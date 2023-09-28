import streamlit as st
# 4. Use streamlit to create a web app
from agent import agent


def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")
    #
    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")
    if query:
        st.write("Doing research for ", query)
        result = agent({"input": query})
        st.info(result['output'])


if __name__ == '__main__':
    main()



