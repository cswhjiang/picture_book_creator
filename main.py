import json
from PIL import Image
import cv2
import requests
import argparse

from openai import OpenAI

client = OpenAI()

image_index = 1

def generate_image(prompt, n:int=1, size:str="1024x1024"):
    global client
    global image_index
    response = client.images.generate(
      model="dall-e-3",
      prompt=prompt,
      size=size,
      quality="standard",
      n=1
    )
    # print('generate_image response:', response)

    image_url = response.data[0].url

    im = Image.open(requests.get(image_url, stream=True).raw)
    im.save("temp_" + str(image_index) + ".png")
    image_index = image_index + 1


    # img = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow('image', img)


    return image_url

def run_conversation(input_prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": input_prompt}]
    print('User:', messages[0]['content'])
    tools = [
            {
              "type": "function",
              "function": {
                "name": "generate_image",
                "description": "generate image by Dall-e 3",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "prompt": {"type": "string", "description": "The prompt to generate image"},
                    "size": {"type": "string", "enum": ['1024x1024', '1024x1792', '1792x1024']}
                  },
                  "required": ["prompt"]
                }
              }
            }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    # message_dict = json.loads(response.model_dump_json())
    # print(message_dict)

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "generate_image": generate_image,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                prompt=function_args.get("prompt"),
                size=function_args.get("size"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        
        return second_response

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Picture Book Creator')
    parser.add_argument('--prompt', type=str,
                        help='your input prompt for creating the book.')
    parser.add_argument('--save_file', type=str,
                        help='your target markdown file to save the created book.')

    args = parser.parse_args()
    # input_prompt = "为小朋友生成一个绘本文章，包含多个图像和多段文字。内容与小动物有关。"

    input_prompt = args.prompt
    save_file = args.save_file
    if not save_file.endswith('.md'):
        save_file = save_file + '.md'



    second_response = run_conversation(input_prompt)
    if second_response is not None:
        message_dict = json.loads(second_response.model_dump_json())
        print('AI:', message_dict['choices'][0]['message']['content'])


        with open(save_file, 'w') as f:
            f.write(message_dict['choices'][0]['message']['content'])
    else:
        print('response from Openai is none, please check')
