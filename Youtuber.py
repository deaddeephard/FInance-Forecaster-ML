import pika
import sys
class Youtuber:
    def __init__(self,youtube_name):
        self.youtube_name=youtube_name
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost',port=5672))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='Youtuber_Requests')

    def publish_video(self,request):
        self.channel.basic_publish(exchange='',routing_key='Youtuber_Requests', body=request)
        print(f"SUCCESS: Video sent to Youtube server.")

if __name__== "__main__":
    youtuber_name= sys.argv[1]
    video_name = " ".join(sys.argv[2:])
    request= f'{{"youtuber": "{youtuber_name}" ,"video": "{video_name}"}}'
    youtuber = Youtuber(youtuber_name)
    youtuber.publish_video(request)










