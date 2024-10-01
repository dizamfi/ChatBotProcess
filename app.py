import os
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from utils import *  # Asegúrate de que utils.py esté en el mismo directorio o en el PYTHONPATH
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ["PINECONE_API_KEY"] = "461faa64-b7ca-4a71-bfe9-bf6f472b905d"

FILE_LIST = "archivos.txt"
OPENAI_API_KEY = "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"
INDEX_NAME = "chatgps"

# Lista de prompts más utilizados (ejemplo)
PROMPTS = [
    "¿Cuál es el objetivo principal de este documento?",
    "¿Qué información clave se destaca en este texto?",
    "¿Cómo se relaciona este contenido con el tema principal?",
    "¿Qué acciones se sugieren en el documento?",
    "¿Cuál es la conclusión principal de este texto?",
    "¿Qué referencias se mencionan en el documento?",
    "¿Cuáles son los puntos débiles del texto?",
    "¿Qué recomendaciones se hacen en el documento?",
    "¿Cómo se estructura el contenido del documento?",
    "¿Qué datos se proporcionan en el documento?",
    # Añadir más prompts según sea necesario
]

st.set_page_config(page_title='GPS Group QuickHelp')
st.header("GPS Group QuickHelp")

# Inicializar las claves en session_state si no están presentes
if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Sidebar with FAQ and Frequently Used Prompts
with st.sidebar:
    archivos = load_name_files(FILE_LIST)
    
    # FAQ Section
    with st.expander("Preguntas Frecuentes"):
        st.write("""
        **¿Cómo funciona esta aplicación?**
        - Esta aplicación permite cargar archivos PDF y hacer preguntas sobre el contenido de esos archivos.

        **¿Qué tipos de archivos se pueden cargar?**
        - Actualmente, solo se admiten archivos PDF.

        **¿Cómo se procesan los archivos?**
        - Los archivos PDF se procesan y se almacenan en un índice para permitir búsquedas eficientes.

        **¿Cómo se realizan las búsquedas?**
        - Las preguntas se buscan en el índice utilizando técnicas de búsqueda por similitud y se generan respuestas basadas en el contenido relevante.
        """)
    
    # Prompt Search Section
    with st.expander("Prompts Más Utilizados"):
        search_term = st.text_input("Buscar prompts:")
        
        # Filtrar prompts que contienen el término de búsqueda
        if search_term:
            filtered_prompts = [prompt for prompt in PROMPTS if search_term.lower() in prompt.lower()]
        else:
            filtered_prompts = PROMPTS
        
        st.write("### Lista de Prompts:")
        
        # Mostrar solo los primeros 3 prompts con scroll vertical si es necesario
        visible_prompts = filtered_prompts[:3]
        if len(filtered_prompts) > 3:
            st.markdown(
                f"""
                <div style="height: 200px; overflow-y: scroll; border-radius: 10px solid #ddd; padding: 5px;">
                    {"<br>".join(f"- {prompt}" for prompt in filtered_prompts)}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("\n".join(f"- {prompt}" for prompt in filtered_prompts))

        # Mostrar el resto en un contenedor con desplazamiento vertical debajo de los primeros 3
        # if len(filtered_prompts) > 3:
        #     st.markdown(
        #         f"""
        #         <div style="height: 200px; overflow-y: scroll; border: 1px solid #ddd; padding: 5px;">
        #             {"<br>".join(f"- {prompt}" for prompt in filtered_prompts[3:])}
        #         </div>
        #         """,
        #         unsafe_allow_html=True
        #     )

    # File uploader and processing
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button('Procesar'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_pinecone(pdf)

        archivos = save_name_files(FILE_LIST, archivos)

    if len(archivos) > 0:
        st.write('Archivos Cargados:')
        lista_documentos = st.empty()
        with lista_documentos.container():
            for arch in archivos:
                st.write(arch)
            if st.button('Borrar Documentos'):
                archivos = []
                clean_files(FILE_LIST)
                lista_documentos.empty()

# Mostrar historial de preguntas y respuestas
st.write("**Historial de Preguntas y Respuestas:**")
if st.session_state['questions']:
    for question, response in zip(st.session_state['questions'], st.session_state['responses']):
        st.markdown(
            f"""
            <div style="background-color:#282828; padding:10px; margin-bottom:10px; border-radius:10px;">
                <b>Pregunta:</b> {question}
            </div>
            <div style="background-color:#0E1117; padding:10px; margin-bottom:10px; border-radius:10px;">
                <b>Respuesta:</b> {response}
            </div>
            """,
            unsafe_allow_html=True
        )

# Formulario de entrada al final
with st.form(key="question_form", clear_on_submit=True):
    user_question = st.text_input("Pregunta: ", key='question_input')
    submit_button = st.form_submit_button(label='Enviar')

    if submit_button and user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Embeddings y vector store están inicializados en utils.py
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vstore = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)

        docs = vstore.similarity_search(user_question, 15)
        llm = ChatOpenAI(model_name='gpt-4o-mini')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.session_state.questions.append(user_question)
        st.session_state.responses.append(respuesta)

        # Actualizar el historial después de enviar
        st.markdown(
            f"""
            <div style="background-color:#282828; padding:10px; margin-bottom:10px; border-radius:10px;">
                <b>Pregunta:</b> {user_question}
            </div>
            <div style="background-color:#0E1117; padding:10px; margin-bottom:10px; border-radius:10px;">
                <b>Respuesta:</b> {respuesta}
            </div>
            """,
            unsafe_allow_html=True
        )


#282828

#0E1117


# FILE_LIST = "archivos.txt"
# OPENAI_API_KEY = "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"

# # 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

# # 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

# st.set_page_config('preguntaDOC')
# st.header("Pregunta a tu PDF")

# with st.sidebar:
#     archivos = load_name_files(FILE_LIST)
#     files_uploaded = st.file_uploader(
#         "Carga tu archivo",
#         type="pdf",
#         accept_multiple_files=True
#     )

#     if st.button('Procesar'):
#         for pdf in files_uploaded:
#             if pdf is not None and pdf.name not in archivos:
#                 archivos.append(pdf.name)
#                 text_to_pinecone(pdf)

#         archivos = save_name_files(FILE_LIST, archivos)

#     if len(archivos) > 0:
#         st.write('Archivos Cargados:')
#         lista_documentos = st.empty()
#         with lista_documentos.container():
#             for arch in archivos:
#                 st.write(arch)
#             if st.button('Borrar Documentos'):
#                 archivos = []
#                 clean_files(FILE_LIST)
#                 lista_documentos.empty()

# if len(archivos) > 0:
#     user_question = st.text_input("Pregunta: ")
#     if user_question:

#         # get openai api key from platform.openai.com
#         os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#         model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
#         # model_name = 'text-embedding-ada-002'

#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#         # embeddings = OpenAIEmbeddings(
#         #     model=model_name,
#         #     openai_api_key=OPENAI_API_KEY
#         # )

#         vstore = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)

#         llm = ChatOpenAI(model_name='gpt-3.5-turbo') # gpt-4o
#         chain = load_qa_chain(llm, chain_type="stuff")

#         docs = vstore.similarity_search(user_question, 3)

#         respuesta = chain.run(input_documents=docs, question=user_question)



#         # OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#         # embeddings = HuggingFaceEmbeddings(
#         #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         # )


#         # Usa el PineconeStore para conectarse al índice
#         # vstore = PineconeStore(index=index, embedding=embeddings, text_key="text")

#         # text_field = "text"

#         # vstore = Pinecone(
#         #     index, embeddings.embed_query, text_field
#         # )

#         # # Obtén el embedding de la pregunta
#         # # query_embedding = embeddings.embed_query(user_question)

#         # # # Realiza la búsqueda en Pinecone
#         # # results = vstore.similarity_search(query_embedding, k=3)
#         # # docs = [res[0] for res in results]

#         # vstore.similarity_search(
#         #     user_question,  # our search query
#         #     k=3  # return 3 most relevant docs
#         # )

#         # llm = ChatOpenAI(model_name='gpt-3.5-turbo')

#         # llm = ChatOpenAI(
#         #     openai_api_key=OPENAI_API_KEY,
#         #     model_name='gpt-3.5-turbo',
#         #     temperature=0.0
#         # )

#         # qa = RetrievalQA.from_chain_type(
#         #     llm=llm,
#         #     chain_type="stuff",
#         #     retriever=vstore.as_retriever()
#         # )

#         # respuesta = qa.run(user_question)

#         # chain = load_qa_chain(llm, chain_type="stuff")
#         # respuesta = chain.run(input_documents=docs, question=user_question)

#         st.write(respuesta)