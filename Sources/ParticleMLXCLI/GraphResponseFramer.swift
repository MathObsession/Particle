import Foundation

/// Public-text view of a channel-framed response. Protocol detection uses the
/// tokenizer's control tokens, never a repository/model name. Tool payloads and
/// private analysis are not exposed or executed by the chat-only graph worker.
struct GraphResponseFramer {
    let start:Int,channel:Int,message:Int,end:Int,stop:Int
    private var header=[Int](),payload=[Int]()
    private var publicFrame=false,inBody=false,analysisFrame=false
    private(set) var analysisTokens=0
    var isInAnalysis:Bool {inBody && analysisFrame}
    private var committed=""
    init?(controls:[String:Int],prompt:[Int],decode:([Int])->String) {
        guard let start=controls["<|start|>"],let channel=controls["<|channel|>"],let message=controls["<|message|>"],let end=controls["<|end|>"],let stop=controls["<|return|>"],Set([start,channel,message,end,stop]).count == 5 else{return nil}
        self.start=start;self.channel=channel;self.message=message;self.end=end;self.stop=stop
        if let index=prompt.lastIndex(of:start) {
            let suffix=Array(prompt.dropFirst(index+1))
            if let boundary=suffix.firstIndex(of:message) {header=Array(suffix.prefix(boundary));setHeader(decode:decode);inBody=true}
            else {header=suffix}
        }
    }
    private mutating func setHeader(decode:([Int])->String) {
        let boundary=header.firstIndex(of:channel),role=decode(Array(header.prefix(boundary ?? header.count)))
        let name=boundary.map{decode(Array(header.dropFirst($0+1)))} ?? "final"
        let channelName=name.split(whereSeparator:{$0.isWhitespace || $0 == "<"}).first.map(String.init) ?? "final"
        analysisFrame=channelName == "analysis"
        publicFrame=role.trimmingCharacters(in:.whitespacesAndNewlines) == "assistant" && ["final","commentary"].contains(channelName) && !role.contains("to=") && !name.contains("to=")
    }
    mutating func append(_ token:Int,decode:([Int])->String)->String {
        if token == start {
            if inBody,publicFrame {committed += decode(payload)}
            header=[];payload=[];inBody=false;publicFrame=false;analysisFrame=false
        } else if !inBody {
            if token == message {setHeader(decode:decode);payload=[];inBody=true}
            else {header.append(token)}
        } else if token == end || token == stop {
            if publicFrame {committed += decode(payload)}
            header=[];payload=[];inBody=false;publicFrame=false;analysisFrame=false
        } else {payload.append(token);if analysisFrame {analysisTokens += 1}}
        return text(decode:decode)
    }
    func finalHeader(encode:(String)->[Int])->[Int] {[end,start]+encode("assistant")+[channel]+encode("final")+[message]}
    static func instantSuffix(controls:[String:Int],prompt:[Int],decode:([Int])->String,encode:(String)->[Int])->[Int] {
        guard let state=Self(controls:controls,prompt:prompt,decode:decode) else{return []}
        if state.inBody {return state.publicFrame ? [] : state.finalHeader(encode:encode)}
        let boundary=state.header.firstIndex(of:state.channel)
        guard decode(Array(state.header.prefix(boundary ?? state.header.count))).trimmingCharacters(in:.whitespacesAndNewlines) == "assistant" else{return []}
        if let boundary {
            let name=decode(Array(state.header.dropFirst(boundary+1))).trimmingCharacters(in:.whitespacesAndNewlines)
            return name == "final" ? [state.message] : [state.message]+state.finalHeader(encode:encode)
        }
        return [state.channel]+encode("final")+[state.message]
    }
    func text(decode:([Int])->String)->String {committed+(inBody && publicFrame ? decode(payload) : "")}
}

/// Gemma-style paired thought blocks. Detection is by tokenizer controls;
/// hidden payloads are never committed to visible chat or historical replies.
struct GraphThoughtFramer {
    let opening:Int,closing:Int
    private(set) var hidden=false
    private(set) var analysisTokens=0
    private var publicTokens=[Int]()
    init?(opening:Int?,closing:Int?,prompt:[Int]) {
        guard let opening,let closing,opening != closing else{return nil}
        self.opening=opening;self.closing=closing
        if let index=prompt.lastIndex(of:opening) {hidden=prompt.lastIndex(of:closing).map{index>$0} ?? true}
    }
    mutating func append(_ token:Int,decode:([Int])->String)->String {
        if token == opening {hidden=true}
        else if token == closing {hidden=false}
        else if hidden {analysisTokens+=1}
        else {publicTokens.append(token)}
        return text(decode:decode)
    }
    func text(decode:([Int])->String)->String {decode(publicTokens)}
}
