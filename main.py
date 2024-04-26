import streamlit as st

from streamlit_option_menu import option_menu

 
import login, analyser
st.set_page_config(
        page_title="Sentiment Analyser",
)


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Sentiment Analyser',
                options=['Login','Analyser'],
                icons=['person-circle','chat-fill',],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "20px"}, 
        "nav-link": {"color":"white","font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == "Login":
            login.app()
        if app == "Analyser":
            analyser.app()    
             
    run()            
         