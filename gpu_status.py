import numpy as np
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import time

def send_mail(subject='No-reply', message='No-reply'):
    email_host = 'smtp.zjnu.edu.cn'  # 服务器地址
    sender = 'wanchang_phd2020@zjnu.edu.cn'  # 发件人
    password = 'wc19880728'  # 密码，如果是授权码就填授权码
    receiver = 'wanchang_phd2020@zjnu.edu.cn'  # 收件人

    msg = MIMEMultipart()
    msg['Subject'] = subject  # 标题
    msg['From'] = ''  # 发件人昵称
    msg['To'] = ''  # 收件人昵称
    mail_msg = '''<p>\n\t {}</p>'''.format(message)
    msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

    # 发送
    smtp = smtplib.SMTP()
    smtp.connect(email_host, 25)
    smtp.login(sender, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
    print('success')

def get_gpu_memory():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu

flag_last = get_gpu_memory()
while True:
    gpu_memory = get_gpu_memory()
    print("gpu free memory:{} ".format(gpu_memory))

    flag =  np.array(gpu_memory)
    num_changed = np.linalg.norm(np.sign(flag - flag_last), ord=1)
    # 如果有一块卡显存改变
    if num_changed > 0:
        while True:
            try:
                send_mail("{} gpu(s) has changed".format(num_changed), "gpu free memory: {} ".format(gpu_memory))
                break
            except:
                print('warning: email not sent.')

    flag_last = flag
    time.sleep(60)
