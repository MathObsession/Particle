#!/usr/bin/env python3
"""Model-free desktop transport fixture. Never imports inference libraries."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
root=Path(os.environ['PARTICLEONE_DATA_PATH'])
def event(kind,**values):
 data=(json.dumps(dict(event=kind,**values),ensure_ascii=False)+'\n').encode()
 # Deliberately split encoded Unicode between reads.
 for start in range(0,len(data),7):sys.stdout.buffer.write(data[start:start+7]);sys.stdout.buffer.flush()
class Stopped(Exception):pass
def interrupted(*_):raise Stopped()
signal.signal(signal.SIGINT,interrupted)
event('ready')
for line in sys.stdin:
 try:
  command=json.loads(line)
  with (root/'commands.jsonl').open('a') as log:log.write(json.dumps(command,ensure_ascii=False)+'\n')
  kind=command['type']
  if kind=='quit':break
  if kind in ['restore','reset']:event('restored',messages=len(command.get('messages') or []));event('ready');continue
  if kind=='agent':
   history=command.get('messages',[]);user=next((m['content'] for m in history if m['role']=='user'),'').split('\n\n<project_context>',1)[0]
   observations=[json.loads(m['content']) for m in history if m['role']=='tool']
   if user=='agent-fixture':
    if not observations:text='<tool_call>read_file<arg_key>path</arg_key><arg_value>main.txt</arg_value></tool_call>'
    elif len(observations)==1:text='<tool_call>edit_file<arg_key>path</arg_key><arg_value>main.txt</arg_value><arg_key>old_string</arg_key><arg_value>color=red</arg_value><arg_key>new_string</arg_key><arg_value>color=green</arg_value></tool_call>'
    elif len(observations)==2:text='<tool_call>run_command<arg_key>command</arg_key><arg_value>printf verified > command-ran.txt; cat main.txt</arg_value></tool_call>'
    else:text='Changed the color to green and verified the file with the approved command.'
   elif user=='agent-needs-verification':
    if not observations:text='<tool_call>write_file<arg_key>path</arg_key><arg_value>verify.txt</arg_value><arg_key>content</arg_key><arg_value>ready</arg_value></tool_call>'
    elif len(observations)==1:text='Saved the file.```python\nprint(42)\n```'
    elif len(observations)==2:text='<tool_call>run_command<arg_key>command</arg_key><arg_value>cat verify.txt</arg_value></tool_call>'
    else:text='Created verify.txt and verified its actual contents.'
   elif not observations and user=='agent-cancel':text='<tool_call>run_command<arg_key>command</arg_key><arg_value>sleep 10</arg_value></tool_call>'
   elif not observations:text='<tool_call>write_file<arg_key>path</arg_key><arg_value>denied.txt</arg_value><arg_key>content</arg_key><arg_value>should not be written</arg_value></tool_call>'
   else:text='The requested operation was denied or blocked; no file was written.'
   event('text',text=text);event('turn',text=text,finish='stop',tokens=10,decode_steps=1,decode_seconds=1,context_used=48);event('ready')
  if kind=='message':
   event('input',utf8_sha256=hashlib.sha256(command['text'].encode()).hexdigest(),utf8_bytes=len(command['text'].encode()))
   event('prefill',completed=10,total=10,reused=0)
   if command['text']=='reasoning-limit':
    event('turn',text='<think>',finish='reasoning_limit',decode_steps=1,decode_seconds=1,context_used=32);event('ready');continue
   if command['text']=='slow':
    event('text',text='working');time.sleep(10)
   text="Hello 🌿\n```python\nprint('नमस्ते')\n```"
   event('text',text=text);event('turn',text=text,finish='stop',tokens=10,decode_steps=1,decode_seconds=1,context_used=32);event('ready')
 except Stopped:event('cancelled');event('ready')
