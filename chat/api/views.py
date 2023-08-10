from django.contrib.auth.models import User
from chat.models import Chat, Message
from rest_framework.decorators import api_view
from .serializers import MessageSerializer
from rest_framework.response import Response
#from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
from django.conf import settings
import os
import json

import torch

import json
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer
openai.api_key = settings.OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "554f8e78-3ce3-42d3-a61e-e3ac03dea22b"
os.environ["PINECONE_ENV"] = "us-west4-gcp-free"

@api_view(['GET'])
def getMessages(request, pk):
    user = User.objects.get(username = request.user.username)
    chat = Chat.objects.filter(user = user)[pk-1]
    messages = Message.objects.filter(chat = chat)
    #context = {'messages' : messages}
    serializer = MessageSerializer(messages, many=True)
    #return render(request, 'chat/chat.html')
    return Response(serializer.data)
    #return HttpResponse(f"You are in {chat.id} chat and you are {user.username}")



# @api_view(['POST'])
@csrf_exempt
def get_prompt_result4(request):
    if request.method == 'POST':
        # Get the prompt from the request body
        data = json.loads(request.body.decode('utf-8'))
        prompt = data.get('prompt')
        user = User.objects.get(username = request.user.username)
        chat = Chat.objects.filter(user = user)[0]
        message = Message(message = prompt, is_user = True, chat = chat)
        message.save()

        
        # model = data.get('model', 'gpt')

        # Check if prompt is present in the request
        if not prompt:
            # Send a 400 status code and a message indicating that the prompt is missing
            return JsonResponse({'error': 'Prompt is missing in the request'}, status=400)

        try:
            # Use the OpenAI SDK to create a completion
            # with the given prompt, model and maximum tokens
            # if model == 'image':
            #     result = openai.create_image(prompt=prompt, response_format='url', size='512x512')
            #     return JsonResponse({'result': result['data'][0]['url']})

            # model_name = 'text-davinci-003' if model == 'gpt' else 'code-davinci-002'
            #max_tokens = 4000
            #prompt = f"Please reply below question in markdown format.\n {prompt}"

            messages_db = Message.objects.filter(chat = chat)
            messages=[
                    {"role": "system", "content": "You are chatbot. Reply all questions in markdown format."},
                    {"role": "user", "content": "Hey! How are you?"},
                    {"role": "assistant",
                     "content": "Hi! I'm fine! What about you?"},
                    {"role": "user", "content": prompt}
                ]
            
            for msg in messages_db:
                role = "user" if msg.is_user else "assistant"
                messages.append({"role" : role, "content" : msg.message})
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages

            )
            message = Message(message = completion.choices[0].message.content, is_user = False, chat = chat)
            message.save()
            # Send the generated text as the response
            return JsonResponse(completion.choices[0].message.content, safe=False)
        except Exception as error:
            error_msg = error.response['error'] if hasattr(
                error, 'response') and error.response else str(error)
            print(error_msg)
            # Send a 500 status code and the error message as the response
            return JsonResponse({'error': error_msg}, status=500)

    # Send a 405 status code if the request method is not POST
    return JsonResponse({'error': 'Method not allowed'}, status=405)


import json
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer

import json
from django.shortcuts import render
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer

import json
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer


@csrf_exempt
def get_prompt_result678(request):
    if request.method == 'POST':
        # Get the input prompt from the request body

        data = json.loads(request.body.decode('utf-8'))
        user_input = data.get('prompt')
        user = User.objects.get(username=request.user.username)
        chat = Chat.objects.filter(user=user)[0]
        message = Message(message=user_input, is_user=True, chat=chat)
        message.save()


        #data = json.loads(request.body.decode('utf-8'))
        #user_input = data.get('user_input')

        # Check if user input is present in the request
        if not user_input:
            return JsonResponse({'error': 'User input is missing in the request'}, status=400)

        try:
            # Initialize tokenizer and model
            #tokenizer = T5Tokenizer.from_pretrained('t5-small')
            #model = T5ForConditionalGeneration.from_pretrained('t5-small')
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            model = T5Tokenizer.from_pretrained("google/flan-t5-base")

            # Load model directly
            #from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            #input_ids = tokenizer("question: How to catch a fish", return_tensors="pt").input_ids
            #outputs = model.generate(input_ids)
            #print(tokenizer.decode(outputs[0], skip_special_tokens=True))






            # Create chat history
            chat_history = [
                {"role": "system", "content": "You are chatting with a chatbot. Please type your messages."},
                {"role": "user", "content": user_input}
            ]

            # Convert chat history to input format for T5
            input_ids = tokenizer.encode("chatbot:", return_tensors='pt')
            chat_input_ids = tokenizer.encode("\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]),
                                              return_tensors='pt')

            input_ids = torch.cat((input_ids, chat_input_ids), dim=1)
            response_ids = model.generate(input_ids)

            chatbot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Update chat history with chatbot response
            chat_history.append({"role": "chatbot", "content": chatbot_response})

            #return JsonResponse({'chatbot_response': chatbot_response, 'chat_history': chat_history})

            return JsonResponse(chatbot_response, safe=False)
        except Exception as error:
            error_msg = str(error)
            print(error_msg)
            return JsonResponse({'error': error_msg}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

@csrf_exempt
def get_prompt_result3 (request):
    if request.method == 'POST':
        # Get the input prompt from the request body
        data = json.loads(request.body.decode('utf-8'))
        user_input = data.get('prompt')
        user = User.objects.get(username=request.user.username)
        chat = Chat.objects.filter(user=user)[0]
        message = Message(message=user_input, is_user=True, chat=chat)
        message.save()

        # Check if user input is present in the request
        if not user_input:
            return JsonResponse({'error': 'User input is missing in the request'}, status=400)

        try:
            # Initialize tokenizer and model
            tokenizer = T5Tokenizer.from_pretrained('t5-small')

            # Load the model from the saved .bin file
            model = T5ForConditionalGeneration.from_pretrained('/Users/tramy/Documents/Projects/chatbot-master/pytorch_model.bin')  # Replace with the actual path

            # Create chat history
            chat_history = [
                {"role": "system", "content": "You are chatting with a chatbot. Please type your messages."},
                {"role": "user", "content": user_input}
            ]

            # Convert chat history to input format for T5
            input_ids = tokenizer.encode("chatbot:", return_tensors='pt')
            chat_input_ids = tokenizer.encode("\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]),
                                              return_tensors='pt')

            input_ids = torch.cat((input_ids, chat_input_ids), dim=1)
            response_ids = model.generate(input_ids)

            chatbot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Update chat history with chatbot response
            chat_history.append({"role": "chatbot", "content": chatbot_response})

            return JsonResponse({'chatbot_response': chatbot_response, 'chat_history': chat_history})
        except Exception as error:
            error_msg = str(error)
            print(error_msg)
            return JsonResponse({'error': error_msg}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
#############################
##########################################################

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

def load_document(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_content = page.get_text()
        metadata = page.metadata
        data.append({'page_content': page_content, 'metadata': metadata})
    doc.close()
    return data
def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        ####
        data = load_document(pdf_path)
        print(f'You have {len(data)} pages in your data')
        print_page_info(data[1])

        chunks = chunk_data(data)

        ##########
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store


def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')


def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history


######

import fitz  # PyMuPDF
import pinecone



def print_page_info(page_data):
    print(f'There are {len(page_data["page_content"])} characters in the page')
    print(page_data['metadata'])

def process_pdf(pdf_path):
    data = load_document(pdf_path)
    print(f'You have {len(data)} pages in your data')
    print_page_info(data[1])

    chunks = chunk_data(data)

    print_embedding_cost(chunks)
    delete_pinecone_index()

    index_name = 'askadocument'
    vector_store = insert_or_fetch_embeddings(index_name)

# Call the function with your PDF file path
pdf_path = 'files/TheThanhToan.pdf'
#process_pdf(pdf_path)

@csrf_exempt
def get_prompt_result(request):
    if request.method == 'POST':
        # Get the input prompt from the request body

        data = json.loads(request.body.decode('utf-8'))
        user_input = data.get('prompt')
        user = User.objects.get(username=request.user.username)
        chat = Chat.objects.filter(user=user)[0]
        message = Message(message=user_input, is_user=True, chat=chat)
        message.save()


        #data = json.loads(request.body.decode('utf-8'))
        #user_input = data.get('user_input')

        # Check if user input is present in the request
        if not user_input:
            return JsonResponse({'error': 'User input is missing in the request'}, status=400)

        try:
            index_name = 'askadocument'
            vector_store = insert_or_fetch_embeddings(index_name)


            answer = ask_and_get_answer(vector_store, user_input)
            #print(answer)

            chatbot_response = answer

            # Update chat history with chatbot response
            #chat_history.append({"role": "chatbot", "content": chatbot_response})

            #return JsonResponse({'chatbot_response': chatbot_response, 'chat_history': chat_history})

            return JsonResponse(chatbot_response, safe=False)
        except Exception as error:
            error_msg = str(error)
            print(error_msg)
            return JsonResponse({'error': error_msg}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
