from petrel_client.client import Client
import io

class MyClient(object):

	def __init__(self):
		self.client = Client("~/petreloss.conf")
	def get(self, key, enable_stream=False):
		index = key.find("/")
		bucket = key[:index]
		key = key[index+1:]
		if bucket == "asr" or bucket ==  "exp":
			return self.client.get("asr:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
		elif bucket == "youtubeBucket" or bucket == "llm-private-datasets":
			return self.client.get("youtube:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
		elif bucket == "llm-private-dataset-snew":
			print("llm:s3://{}/".format(bucket) + key)
			return self.client.get("private:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
	def get_file_iterator(self, key):
		index = key.find("/")
		bucket = key[:index]
		key = key[index+1:]
		if bucket == "asr":
				return self.client.get_file_iterator("asr:s3://{}/".format(bucket) + key)
		elif bucket == "youtubeBucket" or bucket == 'llm-private-datasets':
				return self.client.get_file_iterator("youtube:s3://{}/".format(bucket) + key)
	def put(self, uri, content):
		return self.client.put(uri, content)
	
	def contains(self, key):
		index = key.find("/")
		bucket = key[:index]
		key = key[index+1:]
		if bucket == "asr":
			return self.client.contains("asr:s3://{}/".format(bucket) + key)
		elif bucket == "youtubeBucket" or bucket =='llm-private-datasets':
			return self.client.contains("youtube:s3://{}/".format(bucket) + key)



if __name__ == "__main__":
	url = 'llm-private-dataset-snew/audio_data/audio/7243553921411255098.mp3'
	client = MyClient()
	data = client.get(url)
	with io.BytesIO(data) as fobj:
		print(fobj)

