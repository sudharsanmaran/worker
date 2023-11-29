import os
import pika
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from io import BytesIO

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
    prompt = body.decode()
    print(f"Received prompt: {prompt}")

    # Generate image
    image_data = generate_image(prompt)

    # Send response back to response queue
    ch.basic_publish(
        exchange="",
        routing_key=response_queue,
        body=image_data,
        properties=pika.BasicProperties(correlation_id=properties.correlation_id),
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(f"Sent image for prompt: {prompt}")


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

print("Awaiting RPC requests")
channel.start_consuming()
