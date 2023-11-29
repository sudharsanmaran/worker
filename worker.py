import os
import pika
import torch
from azure.storage.blob import BlobServiceClient
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from io import BytesIO
from dotenv import load_dotenv
import uuid


# Initialize Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float32
).to("cpu")

# RabbitMQ setup
rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")  # Use environment variable
queue_name = os.getenv("REQUEST_QUEUE", "image_requests")  # Use environment variable
response_queue = os.getenv(
    "RESPONSE_QUEUE", "image_responses"
)  # Use environment variable

connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
channel = connection.channel()
channel.queue_declare(queue=queue_name)
channel.queue_declare(queue=response_queue)


load_dotenv()


def upload_image_to_blob(byte_arr, name):
    # Connect to the blob storage account
    connect_str = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_ACCOUNT_KEY')}"

    try:
        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    except Exception as e:
        print(f"An exception occurred while creating BlobServiceClient: {e}")
        return

    # Get the existing container
    container_name = os.getenv("AZURE_CONTAINER_NAME")

    try:
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
    except Exception as e:
        print(f"An exception occurred while getting container client: {e}")
        return

    try:
        # Create a blob client using the blob name
        blob_client = container_client.get_blob_client(name)

        # Upload image data to blob
        blob_client.upload_blob(byte_arr, overwrite=True)

        # Generate image URL
        image_url = blob_client.url

        # Print the image URL
        print(image_url)
        return image_url

    except Exception as e:
        print(f"An exception occurred while uploading the file: {e}")


def generate_image(
    prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5
):
    # Generate an image
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Convert image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


def on_request(ch, method, properties, body):
    # Todo
    # Receive prompt and other properties from request queue
    request_data = body.decode()
    prompt = request_data["prompt"]
    height = request_data["height"]
    width = request_data["width"]
    num_inference_steps = request_data["num_inference_steps"]
    guidance_scale = request_data["guidance_scale"]
    # Generate image

    byte_arr = generate_image(prompt, height, width, num_inference_steps, guidance_scale)

    random_str = str(uuid.uuid4())
    images_name= f"{os.getenv('BASE_NAME')}-{random_str}"

    # Upload image to blob
    image_url = upload_image_to_blob(byte_arr, images_name)

    # Send response back to response queue
    ch.basic_publish(
        exchange="",
        routing_key=response_queue,
        body={"response": image_url},
        properties=pika.BasicProperties(correlation_id=properties.correlation_id),
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(f"Sent image for prompt: {prompt}")


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

print("Awaiting RPC requests")
channel.start_consuming()
