#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header

class mail:
    def __init__(self):

        self.mail_host = "smtp.qq.com"  # 设置服务器
        self.mail_user = "image_caption"  # 用户名
        self.mail_pass = "hymgczleafqufbgj"  # 口令

        self.sender = 'image_caption@qq.com'
        self.receivers = ['image_caption@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    def send(self, title, msg):

        message = MIMEText(msg, 'plain', 'utf-8')
        message['From'] = Header("GPU SERVER", 'utf-8')
        message['To'] = Header("AMDS", 'utf-8')

        subject = title
        message['Subject'] = Header(subject, 'utf-8')

        try:
            smtpObj = smtplib.SMTP_SSL()
            smtpObj.connect(self.mail_host, 465)  # 25 为 SMTP 端口号
            smtpObj.login(self.mail_user, self.mail_pass)
            smtpObj.sendmail(self.sender, self.receivers, message.as_string())
            smtpObj.close()
            print('send email success')
        except smtplib.SMTPException,e:
            print(e)
            print('send email failed')