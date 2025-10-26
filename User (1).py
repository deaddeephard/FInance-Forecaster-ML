import pika
import sys
class User:
    def __init__(self,username):
        self.username=username
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost',port=5672))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="User_Requests")
        self.channel.queue_declare(queue=self.username)
            
    def update_subscription(self,request):
        self.channel.basic_publish(exchange='',routing_key='User_Requests',body=request)
        print("SUCCESS: Subscription/Login request sent.")
    
    def receive_notifications(self):
        def callback2(ch, method, properties, body):
            notification = body.decode()
            print(f"{notification}")    
        self.channel.basic_consume(queue=self.username, on_message_callback=callback2, auto_ack=True)
        print("User is receiving notifications. To exit press CTRL+C")
    
    def start_consuming(self,request):
        self.update_subscription(request)
        self.receive_notifications()
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print(" User Logged out.")
            self.connection.close()

if __name__== "__main__":
    if len(sys.argv)==2:
        username = sys.argv[1]
        request= f'{{"user":"{username}"}}'

    elif len(sys.argv)==4:
        username= sys.argv[1]
        youtuber_name= sys.argv[3]
        if(str(sys.argv[2])=="s"):
            request = f'{{"user": "{username}","youtuber": "{youtuber_name}","subscribe": "True"}}'
        elif(str(sys.argv[2])=="u"):
            request = f'{{"user": "{username}","youtuber": "{youtuber_name}","subscribe": "False"}}'
    
    user = User(username)
    user.start_consuming(request)


    



    

    


    



