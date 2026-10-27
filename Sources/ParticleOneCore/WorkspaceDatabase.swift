import Foundation
import CSQLite

public struct WorkspaceSummary: Identifiable, Sendable {
    public let id: UUID
    public let name: String
}
/// One private SQLite database contains independent workspace records. Each
/// atomic record includes its settings, chats, message IDs, and saved memories.
/// No network service or external account is involved.
public enum WorkspacePersistence {
    public static var defaultDirectory: URL {FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/ParticleOne",isDirectory:true)}
    public static func load(from directory: URL) throws -> SavedWorkspace {
        let dbURL=directory.appendingPathComponent("particleone.sqlite")
        if !FileManager.default.fileExists(atPath:dbURL.path) {
            let legacy=directory.appendingPathComponent("workspace.json")
            let value=FileManager.default.fileExists(atPath:legacy.path) ? try decode(Data(contentsOf:legacy)) : SavedWorkspace()
            try save(value,to:directory) // The original JSON is deliberately retained.
            return value
        }
        let db=try Database(directory);defer{db.close()}
        guard let data=try db.blob("SELECT payload FROM workspaces WHERE id=(SELECT value FROM metadata WHERE key='active')") else {throw CocoaError(.coderReadCorrupt)}
        return try decode(data)
    }
    public static func save(_ value: SavedWorkspace,to directory: URL) throws {
        guard value.version==1,value.settings.valid,value.name.count<=80,value.username.count<=80,value.memories.count<=100,value.memories.allSatisfy({!$0.text.isEmpty && $0.text.count<=2000}) else {throw CocoaError(.coderInvalidValue)}
        let db=try Database(directory);defer{db.close()}
        try db.exec("BEGIN IMMEDIATE")
        do {
            try db.write("INSERT INTO workspaces(id,name,payload) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,payload=excluded.payload",texts:[value.id.uuidString,value.name],data:JSONEncoder().encode(value))
            try db.write("INSERT INTO metadata(key,value) VALUES('active',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",texts:[value.id.uuidString])
            try db.exec("COMMIT")
        } catch {try? db.exec("ROLLBACK");throw error}
    }
    public static func list(in directory: URL) throws -> [WorkspaceSummary] {
        let db=try Database(directory);defer{db.close()}
        let rows=try db.rows("SELECT id,name FROM workspaces ORDER BY name COLLATE NOCASE")
        return try rows.map {row in guard let id=UUID(uuidString:row[0]) else{throw CocoaError(.coderReadCorrupt)};return WorkspaceSummary(id:id,name:row[1])}
    }
    public static func open(_ id: UUID,in directory: URL) throws -> SavedWorkspace {
        let db=try Database(directory);defer{db.close()}
        guard let data=try db.blob("SELECT payload FROM workspaces WHERE id=?",text:id.uuidString) else{throw CocoaError(.fileNoSuchFile)}
        let value=try decode(data)
        try db.write("INSERT INTO metadata(key,value) VALUES('active',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",texts:[id.uuidString])
        return value
    }
    private static func decode(_ data: Data) throws -> SavedWorkspace {
        let value=try JSONDecoder().decode(SavedWorkspace.self,from:data)
        guard value.version==1,value.settings.valid,value.memories.count<=100,value.memories.allSatisfy({!$0.text.isEmpty && $0.text.count<=2000}) else{throw CocoaError(.coderReadCorrupt)}
        return value
    }
}
private final class Database {
    private var handle: OpaquePointer?
    private let transient=unsafeBitCast(-1,to:sqlite3_destructor_type.self)
    init(_ directory: URL) throws {
        let fm=FileManager.default
        if fm.fileExists(atPath:directory.path),try directory.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink==true {throw CocoaError(.fileWriteNoPermission)}
        try fm.createDirectory(at:directory,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let file=directory.appendingPathComponent("particleone.sqlite")
        if fm.fileExists(atPath:file.path) {
            guard try file.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink != true else{throw CocoaError(.fileWriteNoPermission)}
        } else {guard fm.createFile(atPath:file.path,contents:nil,attributes:[.posixPermissions:0o600]) else{throw CocoaError(.fileWriteUnknown)}}
        try fm.setAttributes([.posixPermissions:0o600],ofItemAtPath:file.path)
        guard sqlite3_open_v2(file.path,&handle,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_FULLMUTEX,nil)==SQLITE_OK else{close();throw CocoaError(.fileReadCorruptFile)}
        sqlite3_busy_timeout(handle,3000)
        do {try exec("PRAGMA secure_delete=ON; PRAGMA journal_mode=DELETE; CREATE TABLE IF NOT EXISTS workspaces(id TEXT PRIMARY KEY,name TEXT NOT NULL,payload BLOB NOT NULL); CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);")}
        catch {close();throw error}
    }
    func close(){if let handle{sqlite3_close(handle);self.handle=nil}}
    deinit {close()}
    func exec(_ sql: String) throws {guard sqlite3_exec(handle,sql,nil,nil,nil)==SQLITE_OK else{throw CocoaError(.fileWriteUnknown)}}
    private func prepare(_ sql: String) throws -> OpaquePointer {
        var statement:OpaquePointer?
        guard sqlite3_prepare_v2(handle,sql,-1,&statement,nil)==SQLITE_OK,let statement else{throw CocoaError(.fileReadCorruptFile)}
        return statement
    }
    func write(_ sql: String,texts:[String],data:Data?=nil) throws {
        let statement=try prepare(sql);defer{sqlite3_finalize(statement)}
        for (index,text) in texts.enumerated(){guard sqlite3_bind_text(statement,Int32(index+1),text,-1,transient)==SQLITE_OK else{throw CocoaError(.fileWriteUnknown)}}
        if let data {let result=data.withUnsafeBytes{sqlite3_bind_blob(statement,Int32(texts.count+1),$0.baseAddress,Int32(data.count),transient)};guard result==SQLITE_OK else{throw CocoaError(.fileWriteUnknown)}}
        guard sqlite3_step(statement)==SQLITE_DONE else{throw CocoaError(.fileWriteUnknown)}
    }
    func blob(_ sql:String,text:String?=nil) throws -> Data? {
        let statement=try prepare(sql);defer{sqlite3_finalize(statement)}
        if let text{sqlite3_bind_text(statement,1,text,-1,transient)}
        let status=sqlite3_step(statement)
        if status==SQLITE_DONE{return nil}
        guard status==SQLITE_ROW,let bytes=sqlite3_column_blob(statement,0) else{throw CocoaError(.fileReadCorruptFile)}
        return Data(bytes:bytes,count:Int(sqlite3_column_bytes(statement,0)))
    }
    func rows(_ sql:String) throws -> [[String]] {
        let statement=try prepare(sql);defer{sqlite3_finalize(statement)};var result=[[String]]()
        while true {
            let status=sqlite3_step(statement);if status==SQLITE_DONE{return result}
            guard status==SQLITE_ROW,let a=sqlite3_column_text(statement,0),let b=sqlite3_column_text(statement,1) else{throw CocoaError(.fileReadCorruptFile)}
            result.append([String(cString:a),String(cString:b)])
        }
    }
}
