import Foundation

private final class AgentWebRedirect:NSObject,URLSessionTaskDelegate,@unchecked Sendable {
    func urlSession(_ session:URLSession,task:URLSessionTask,willPerformHTTPRedirection response:HTTPURLResponse,newRequest request:URLRequest,completionHandler:@escaping(URLRequest?)->Void){completionHandler(request.url?.scheme=="https" && ["mcp.exa.ai","r.jina.ai"].contains(request.url?.host ?? "") ? request : nil)}
}
/// Only the fixed public search provider is contacted, never a model-selected
/// local/private network endpoint. Queries/URLs require approval in the host.
public actor AgentWeb {
    private var sessionID:String?
    private var available=Set<String>()
    private var searchFields=Set<String>()
    private let session:URLSession
    public init(){let config=URLSessionConfiguration.ephemeral;config.httpShouldSetCookies=false;config.urlCache=nil;config.timeoutIntervalForRequest=30;config.timeoutIntervalForResource=45;session=URLSession(configuration:config,delegate:AgentWebRedirect(),delegateQueue:nil)}
    public static func publicURL(_ value:String)throws->URL{
        guard let url=URL(string:value),url.scheme=="https",let host=url.host?.lowercased(),url.user==nil,url.password==nil,value.utf8.count<=2048,
              host.contains("."),!host.contains(":"),host.first?.isNumber != true,!host.hasSuffix(".local"),!host.hasSuffix(".internal"),!host.hasSuffix(".localhost"),!host.hasSuffix(".test"),host != "localhost",!host.hasSuffix(".home"),!host.hasSuffix(".arpa"),url.port==nil || url.port==443 else{throw AgentFailure.message("Use a public HTTPS page without credentials or a custom port.")}
        return url
    }
    private func request(_ object:[String:Any])async throws->[String:Any]{
        var request=URLRequest(url:URL(string:"https://mcp.exa.ai/mcp")!);request.httpMethod="POST";request.httpBody=try JSONSerialization.data(withJSONObject:object)
        request.setValue("Particle/0.3",forHTTPHeaderField:"User-Agent");request.setValue("application/json",forHTTPHeaderField:"Content-Type");request.setValue("application/json, text/event-stream",forHTTPHeaderField:"Accept")
        if let sessionID{request.setValue(sessionID,forHTTPHeaderField:"Mcp-Session-Id")}
        let (stream,response)=try await session.bytes(for:request)
        guard let http=response as? HTTPURLResponse,(200...299).contains(http.statusCode) else{throw AgentFailure.message("Web provider unavailable or rate limited. Try later; no search result was fabricated.")}
        if let id=http.value(forHTTPHeaderField:"Mcp-Session-Id"){sessionID=id}
        var data=Data();for try await byte in stream {try Task.checkCancellation();guard data.count<1_048_576 else{throw AgentFailure.message("Web response exceeded 1 MiB.")};data.append(byte)}
        if data.isEmpty{return [:]}
        var object=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
        if object==nil{for line in String(decoding:data,as:UTF8.self).components(separatedBy:"\n") where line.hasPrefix("data:"){if let decoded=(try? JSONSerialization.jsonObject(with:Data(line.dropFirst(5).utf8))) as? [String:Any]{object=decoded}}}
        guard let object,object["error"]==nil else{throw AgentFailure.message("Web provider rejected the request. Retry later or use another source.")};return object
    }
    private func reader(_ value:String)async throws->String{
        _=try Self.publicURL(value)
        guard let url=URL(string:"https://r.jina.ai/"+value) else{throw AgentFailure.message("Invalid public URL.")}
        var request=URLRequest(url:url);request.setValue("Particle/0.3",forHTTPHeaderField:"User-Agent");request.setValue("markdown",forHTTPHeaderField:"X-Return-Format")
        let (stream,response)=try await session.bytes(for:request)
        guard let response=response as? HTTPURLResponse,(200...299).contains(response.statusCode) else{throw AgentFailure.message("Both public page readers could not fetch this URL. The page may restrict access.")}
        var data=Data();for try await byte in stream{try Task.checkCancellation();guard data.count<1_048_576 else{throw AgentFailure.message("Page exceeded 1 MiB; use a narrower source.")};data.append(byte)}
        guard !data.isEmpty else{throw AgentFailure.message("Page reader returned no content.")}
        return "Public page reader: Jina\n"+String(String(decoding:data,as:UTF8.self).prefix(8000))
    }
    public func run(_ call:AgentCall)async throws->String{
        if available.isEmpty {
            let list=try await request(["jsonrpc":"2.0","id":1,"method":"tools/list"])
            guard let result=list["result"] as? [String:Any],let tools=result["tools"] as? [[String:Any]] else{throw AgentFailure.message("Web provider did not advertise its tools.")}
            available=Set(tools.compactMap{$0["name"] as? String})
            if let search=tools.first(where:{$0["name"] as? String=="web_search_exa"}),let schema=search["inputSchema"] as? [String:Any],let properties=schema["properties"] as? [String:Any]{searchFields=Set(properties.keys)}
        }
        let name:String
        var args=[String:Any]()
        if call.name=="web_search",let query=call.arguments["query"],!query.isEmpty,query.count<=512,available.contains("web_search_exa"){
            name="web_search_exa";args["query"]=query
            if searchFields.contains("numResults"){args["numResults"]=5}
            if searchFields.contains("objective"){args["objective"]="Find relevant primary documentation and trustworthy sources, with source URLs and useful excerpts."}
        }else if call.name=="fetch_url",let value=call.arguments["url"]{
            _=try Self.publicURL(value)
            if available.contains("web_fetch_exa"){name="web_fetch_exa";args=["urls":[value],"maxCharacters":6000]}
            else if available.contains("crawling_exa"){name="crawling_exa";args=["url":value,"maxCharacters":6000]}
            else{throw AgentFailure.message("This web provider does not offer page fetching.")}
        }else{throw AgentFailure.message("Invalid or unavailable web query.")}
        let response=try await request(["jsonrpc":"2.0","id":2,"method":"tools/call","params":["name":name,"arguments":args]])
        guard let result=response["result"] as? [String:Any],let content=result["content"] as? [[String:Any]] else{throw AgentFailure.message("Web tool returned an invalid response.")}
        if (result["isError"] as? Bool)==true{if call.name=="fetch_url",let url=call.arguments["url"]{return try await reader(url)};throw AgentFailure.message("Web provider: "+String(content.compactMap{$0["text"] as? String}.joined(separator:"\n").prefix(1000)))}
        return String(content.compactMap{$0["text"] as? String}.joined(separator:"\n").prefix(8000))
    }
}
