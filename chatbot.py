import itchat, time, re
from itchat.content import *
import urllib2,urllib
import json

@itchat.msg_register(TEXT)
def text_reply(msg):
    info = msg['Text'].encode('UTF-8')
    url = 'http://www.tuling123.com/openapi/api'
    data = {'key':'0ca3121447deed5f7848b5c1544d80ed', 'info':info, 'userid':msg.FromUserName}
    data = urllib.urlencode(data)

    url2 = urllib2.Request(url, data)
    response = urllib2.urlopen(url2)

    apicontent = response.read()
    s = json.loads(apicontent, encoding='utf-8')
    print(s)
    if s['code'] == 100000:
        return s['text']

itchat.auto_login(enableCmdQR=2,hotReload=True)
itchat.run()
