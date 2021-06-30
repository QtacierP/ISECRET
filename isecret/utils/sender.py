# -*- coding: utf-8 -*-
import smtplib
from email.mime.text import MIMEText
from email.header import Header


class EMailSender(object):
    def __init__(self, send_addr, user, password, host, max_num_try=10):
        self.addr = send_addr
        self.user = user
        self.password = password
        self.max_num_try = max_num_try
        self.host = host
        self.connect = False
        self.log = False
        self._init_connect()
        self._log()

    def _init_connect(self):
        self._sender = smtplib.SMTP()
        num = 0
        while True:
            if num == self.max_num_try:
                print('Cannot connect to server {}'.format(self.host))
                return
            if self._sender.connect(self.host, 25):
                break
            num += 1
        print('Connect successfully to {}!'.format(self.host))
        self.connect = True

    def _log(self):
        try:
            self._sender.login(self.user, self.password)
            self.log = True
            print('Login successfully with {}!'.format(self.user))
        except Exception as e:
            print('Cannot login ! Please check your user/password !')
            return

    def send(self, subject, msg, rev_addr):
        # Only support English msg
        if not self.connect:
            print('Please first connect to one host!')
            return
        if not self.log:
            print('Please login your account!')
            return
        message = MIMEText(msg, 'plain')
        message['From'] = Header(self.addr)
        message['To'] = Header(rev_addr)
        message['Subject'] = Header(subject)
        try:
            self._sender.sendmail(self.addr, rev_addr, message.as_string())
        except Exception as e:
            print('Cannot send E-mail!')
            print(e)
            return


