import os
import sys
import time
from typing import Optional
import argparse
import datetime


import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex

from ds_rag_framework.agents.chat import ChatAgent
from ds_rag_framework.llms.prompts import PROMPTS_DICT
from ds_rag_framework.utils import load_config_json


class MainChatConfig:
    def __init__(
        self,
        config_file_path: str = None
    ) -> None:
        # remove deploy button
        st.markdown(
            r"""
            <script>
                var elems = window.parent.document.querySelectorAll('p');
                elem.style.fontSize = '20px';
            </script>
            <style>
            .stDeployButton {
                visibility: hidden;   
            }
            h2 {text-align: center;}
            </style>
            """, unsafe_allow_html=True
        )
        if config_file_path:
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(f"Config file {config_file_path} does not exist.")
            self.config_file_path = config_file_path
            self.initiate_app()



    def initiate_app(self):
        st.title("HealthEdge Chatbot Interface!")

        if "vector_index" in st.session_state:
            print(f"building w/ existing vector index {st.session_state.vector_index}")
            st.session_state.agent = self.initiate_chat_agent(vector_index=st.session_state.vector_index)
        if "agent" not in st.session_state:
            st.session_state.agent = self.initiate_chat_agent()
        self.display_chat_messages()

    def initiate_chat_agent(
        self,
        system_prompt: Optional[str] = None,
        vector_index: Optional[VectorStoreIndex] = None
    ) -> 'ChatAgent':
        config = load_config_json(self.config_file_path)
        param_dict = config['chat_agent'] | config['llm_settings']
        param_dict['system_prompt'] = " ".join(config["chat_agent"]["system_prompt"]) if not system_prompt else system_prompt
        location = param_dict['vector_store_location']
        if location not in ['local', 'cogsearch', 'vertex']:
            raise ValueError("Invalid vector store location. Must be either 'local', 'cogsearch', or 'vertex'.")

        if vector_index:
            param_dict.pop('vector_store_location')
            param_dict.pop('index_path')
            agent = ChatAgent.from_local_storage(vector_index=vector_index, **param_dict)
        elif location == 'local':
            param_dict.pop('vector_store_location')
            agent = ChatAgent.from_local_storage(**param_dict)
        elif location == 'cogsearch':
            param_dict.pop('vector_store_location')
            agent = ChatAgent.from_cog_search(**param_dict)
        elif location == 'vertex':
            param_dict.pop('vector_store_location')
            agent = ChatAgent.from_vertex_ai(**param_dict)
        return agent

    def display_chat_messages(self):
        def helper_stream_time_control(response_gen):
            for word in response_gen:
                yield word
                time.sleep(0.07)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.chat_memory.append(ChatMessage.from_str(prompt, "user"))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner('Generating response...'):
                    res = st.session_state.agent.query(
                        prompt,
                        chat_history=st.session_state.chat_memory,
                        stream=True
                    )
                st.success('Done!')
                response = st.write_stream(helper_stream_time_control(res.response_gen))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_memory.append(ChatMessage.from_str(response, "assistant"))
    
class SidebarConfig:
    def __init__(
        self,
        tab_names: list[str] = ["Tone", "Prompt", "Data Filters"]   
    ) -> None:
        with st.sidebar:
            st.header("Chatbot Configuration")
            tabs = st.tabs(tab_names)
            for tab, tab_name in zip(tabs, tab_names):
                with tab:
                    if tab_name == "Tone":
                        self.config_tone()
                    elif tab_name == "Prompt":
                        self.config_prompt()
                    elif tab_name == "Data Filters":
                        self.config_data_filters()

    def config_tone(self):
        with st.form(key="config_tone_form"):
            st.write("Select a default tone or Customize your chatbot's tone here!")
            container = st.container()
            tone_options = container.selectbox(
                label="Default Tones",
                index=None,
                placeholder="Select default tone...",
                options=['Formal', 'Informal', 'Social Media'],
                # on_change=lambda: update_tone(container),
                key="selected_tone_option"
            )
            submitted = st.form_submit_button("Update")
            if submitted:
                with st.spinner('Updating Chatbot Prompt...'):
                    pass
                st.success("Successfully updated chatbot prompt!")

    def config_prompt(self):
        st.write("Select a default prompt or Customize your chatbot's prompt here!")
        def helper_update_prompt(container):
            if st.session_state.selected_prompt_option:
                container.text_area(
                    label="Selected prompt: ",
                    value=PROMPTS_DICT[st.session_state.selected_prompt_option],
                    height=600
                )


        container = st.container()
        prompt_options = container.selectbox(
            label="Default Prompts",
            index=None,
            placeholder="Select default prompt...",
            options=self.get_existing_default_prompts(),
            on_change=lambda: helper_update_prompt(container),
            key="selected_prompt_option"
        )

        with st.form(key="config_prompt_form"):
            if not st.session_state.selected_prompt_option:
                system_prompt = st.text_area(
                    "Custom Prompt (Instruction)",
                    key="system_prompt",
                    placeholder="Ex: You are an AI assistant trained to provide informative and accurate responses by consulting a comprehensive database of technical documents. Your goal is to assist users by answering their questions with detailed explanations, relevant examples, and up-to-date information. Always ensure your responses are clear, concise, and directly address the user's query. ",
                    height=600
                )

            submitted = st.form_submit_button("Update")
            if submitted:
                with st.spinner('Updating Chatbot Prompt...'):
                    if st.session_state.selected_prompt_option:
                        system_prompt = PROMPTS_DICT[st.session_state.selected_prompt_option]
                    if "vector_index" in st.session_state:
                        print(f"building w/ existing vector index {st.session_state.vector_index}")
                        st.session_state.agent = ChatAgent.from_local_storage(vector_index=st.session_state.vector_index,
                                                                                system_prompt=system_prompt)
                    else:
                        st.session_state.agent = MainChatConfig(config_file_path=st.secrets["CONFIG_FILE_PATH"]).initiate_chat_agent(system_prompt=system_prompt)
                st.success("Successfully updated chatbot prompt!")

    def get_existing_default_prompts(self):
        default_prompts = []
        for key, value in PROMPTS_DICT.items():
            if str(key).startswith("DEFAULT"):
                default_prompts.append(str(key))
        return default_prompts

    def config_data_filters(self):
        st.write("Select metadata filters here!")
        st.selectbox("Filter by data type: ", options=["Social Media Posts", "FAQs", "Website Content", "Customer profiles", "Insights CSV Data"])
        today = datetime.datetime.now()
        jan_1 = datetime.date(today.year, 1, 1)
        dec_31 = datetime.date(today.year, 12, 31)

        d = st.date_input(
            "Filter by date of upload: ",
            (jan_1, today),
            jan_1,
            dec_31,
            format="MM.DD.YYYY",
        )

if __name__ == '__main__':
    MainChatConfig(config_file_path=st.secrets["CONFIG_FILE_PATH"])
    
    SidebarConfig()