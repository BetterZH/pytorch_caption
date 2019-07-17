import itchat
from itchat.content import *

class weixin:
    def __init__(self):
        self.nickName = 'AMDS'

    def login(self):
        itchat.auto_login(True, enableCmdQR=2)

    def send(self,msg):
        try:
            authors = itchat.search_friends(nickName=self.nickName)
            if len(authors) > 0:
                authors[0].send(msg)
            print('send weixin success')
        except:
            print('send weixin failed')