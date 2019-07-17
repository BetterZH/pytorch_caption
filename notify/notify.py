import mail
import weixin

class notify:
    def __init__(self):
        self.use_weixin = False
        self.mail = mail.mail()
        if self.use_weixin:
            self.weixin = weixin.weixin()

    def login(self):
        if self.use_weixin:
            self.weixin.login()

    def send_wx(self, msg):
        try:
            if self.use_weixin:
                self.weixin.send(msg)
        except:
            print('send msg failed')

    def send(self, title, msg):
        try:
            self.mail.send(title, msg)
            if self.use_weixin:
                self.weixin.send(msg)
        except:
            print('send msg failed')
