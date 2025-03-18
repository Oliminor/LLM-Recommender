
import os
import openai
import pickle
import streamlit as st
import torch

from urllib.error import URLError
from redisvl.vectorize.text import HFTextVectorizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

from app.config import (
    CHAT_MODEL,
    VECTORIZER,
    DATAFILE
)

from app.retrieve import (
    create_retrieval_index,
    retrieve_context,
    make_filter,
    retrieve_top_three_hotels
)
from app.prompt import (
    make_prompt,
    generate_hyde_prompt,
    format_prompt_reviews,
    get_recommended_hotel_prompt
)
from junk import (
    insert_comment,
    insert_book,
    search_book
)

from Article_Recommender import (
    add_article,
    get_user_id,
    get_user_articles,
    search_similar_articles,
    vectorize_all_articles,

)

from app.constants import (
    STATES,
    CITIES
)

def recommend_hotel(positive, negative, reviews):

    prompt = make_prompt(positive, negative, reviews)
    retrieval = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{'role':"user",
                   'content': prompt}],
        max_tokens=1000)

    # get the response
    response = retrieval['choices'][0]['message']['content']
    return response


def get_hotel_name(recommendation):
    prompt = get_recommended_hotel_prompt(recommendation)
    retrieval = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{'role':"user",
                   'content': prompt}],
        max_tokens=1000)

    # get the response
    response = retrieval['choices'][0]['message']['content']
    return response

@st.cache_resource
def vectorizer():
    return HFTextVectorizer(f"sentence-transformers/{VECTORIZER}")

@st.cache_data
def load_data():
    data = pickle.load(open(DATAFILE, "rb"))
    return data

def set_city():
    state = st.session_state["state"]
    try:
        return CITIES[state][0]
    except IndexError:
        return []
    
import app.state as state


def main(): 
    vectorize_all_articles()


    # Streamlit UI
    st.sidebar.title("User Panel")

    # Input for username at the top of the sidebar
    username = st.sidebar.text_input("Enter your username")

    # Button to toggle article submission panel
    if st.sidebar.button("➕ Add New Article"):
        state.ArticleState = state.ArticleState.ADD_ARTICLE

    if st.sidebar.button("Recommend Articles"):
        state.ArticleState = state.ArticleState.RECOMMEND_ARTICLE

    # Article submission form in the main area
    if state.ArticleState == state.ArticleState.ADD_ARTICLE:
        st.title("📝 Add a New Article")

        article_title = st.text_input("Article Title")
        article_body = st.text_area("Article Body")

        if st.button("Submit Article"):
            if username and article_title and article_body:
                message = add_article(username, article_title, article_body)
                st.success(message)
                st.session_state["show_form"] = False  # Hide form after submission
            else:
                st.error("❌ Please fill in all fields before submitting.")

    # Article recommender
    if state.ArticleState == state.ArticleState.RECOMMEND_ARTICLE:
        if st.button("Recommend Article"):
            user = get_user_id(username)
            if not user:
                st.write("❌ Username not found")  
            else:    
                articles = get_user_articles(user)
                if not articles:
                    st.write("❌ Article(s) not found")
                else:
                    # Iterate over each article and fetch similar articles
                    for article in articles:
                        article_id, article_title = article  # Unpack the tuple into id and title

                        # Print the article title (optional)
                        st.write(f"Article title: {article_title}")

                        # Use the article_id in the search_similar_articles function
                        similar_articles = search_similar_articles(article_id, user)

                        # Display the similar articles
                        for similar_article in similar_articles:
                            st.write(f"Similar article title: {similar_article[1]}")

''' #Book pratice code
    st.write("# Bence's Book practice")

    st.write("## Search for a book")
    search_title = st.text_input("Enter book title to search")

    # Search button
    if st.button("Search"):
        state.CURRENT_STATE = state.BookAppState.DEFAULT
        if search_title:
            print("Search")
            books = search_book(search_title)
            if books:
                print("Books")
                state.SELECTED_BOOKS_ID = books
                state.CURRENT_STATE = state.BookAppState.IS_BOOK_FOUND
            else:
                print("Else")
                state.CURRENT_STATE = state.BookAppState.IS_NOT_FOUND


        else:
            st.warning("Please enter a title to search.")

    if state.CURRENT_STATE == state.BookAppState.IS_BOOK_FOUND:
        for book in state.SELECTED_BOOKS_ID:
            col1, col2 = st.columns([3, 1])  # This defines the width ratio (3:1)

            # In the first column, write some text
            with col1:
                st.write(f"📖 **Title:** {book[1]}")

            # In the second column, place the text input box
            with col2:
                if st.button("Add Comment", key=f"add_comment_{book}"):
                    state.CURRENT_STATE = state.BookAppState.IS_BOOK_SELECTED
                    state.SELECTED_BOOK_ID = book
                    st.rerun()

    if state.CURRENT_STATE == state.BookAppState.IS_BOOK_SELECTED:
        st.write(f"## Add Comment to the book: {state.SELECTED_BOOK_ID[1]}")
        book_comment = st.text_input("Comment")
        if st.button("Add Comment"):
            insert_comment(state.SELECTED_BOOK_ID[0], book_comment)
            st.rerun()

    if state.CURRENT_STATE == state.BookAppState.IS_NOT_FOUND:
        st.write("No books found with that title.")
        st.write("## Add a book to the library!")
        book_title = st.text_input("Book Title")
        book_comment = st.text_input("Comment")
        if st.button("Add to the library"):
            if book_title and book_comment:  # Ensure fields are not empty
                insert_book(book_title, book_comment)
            else:
                st.warning("Please fill in both fields before adding.")
'''

''' # Hotel codes
    # this data thing stores a list of dictionaries.
    data = load_data()
    print(type(data))
    # for datum in data:
    #     print(datum)
    INDEX = create_retrieval_index(data)
    EMBEDDING_MODEL = vectorizer()

    try:
        # Defining default values
        defaults = {
            "state": "",
            "city": "",
            "positive": "",
            "negative": "",
            "response": "",
            "hotel_info": "",
            "hotel_reviews": "",
            "all_similar_reviews": ""
        }

        # Checking if keys exist in session state, if not, initializing them
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        col1, col2 = st.columns([4,2])

        st.write("# Bence's LLM Hotel Recommender")
        with st.sidebar:
            st.write("## Filter By Location")
            st.selectbox("State", STATES, key="state", on_change=set_city)
            st.selectbox("City", CITIES[st.session_state['state']], key="city")

        st.write("The LLM Hotel Recommender is a Streamlit app that uses Redis and the " +
                 "OpenAI API to generate hotel recommendations based on a user's preferences.")


        st.text_input("What would you like in a hotel?", key="positive")
        st.text_input("What would you like to avoid in a hotel?", key="negative")

        if st.button("Find Hotel"):
            with st.spinner("OpenAI and Redis are working to find you a hotel..."):
                # filter
                query_filter = make_filter(st.session_state['state'], st.session_state['city'])

                # make a hyde prompt
                hyde_prompt = generate_hyde_prompt(
                    st.session_state['positive'],
                    st.session_state['negative']
                )
                print(f"Prompt: {hyde_prompt}")
                print(f"Embed: {EMBEDDING_MODEL.embed(hyde_prompt)}")
                # Retrieve the context
                context = retrieve_context(INDEX,
                                           hyde_prompt,
                                           EMBEDDING_MODEL,
                                           query_filter=query_filter)
                top_three_hotels = retrieve_top_three_hotels(context)

                # TODO catch index error
                top_hotel = top_three_hotels[0]
                top_hotel_reviews = format_prompt_reviews([top_hotel])
                other_options = format_prompt_reviews(top_three_hotels)

                recommendation = recommend_hotel(
                    st.session_state['positive'],
                    st.session_state['negative'],
                    top_hotel_reviews
                )

                hotel_info = {
                    "Hotel Name": top_hotel['name'],
                    "Hotel Address": top_hotel['address'],
                    "City": top_hotel['city'],
                    "State": top_hotel['state'],
                }
                hotel_info = "\n" + "\n".join([f"{k}: {v}" for k, v in hotel_info.items()])
                st.session_state['response'] = recommendation
                st.session_state['hotel_info'] = hotel_info
                st.session_state['hotel_reviews'] = top_hotel_reviews
                st.session_state['all_similar_reviews'] = other_options


            st.write("### Recommendations")
            st.write(f"{st.session_state['response']}")
            with st.expander("Show Hotel Info"):
                st.text(st.session_state['hotel_info'])
            with st.expander("Show Hotel Reviews"):
                st.text(st.session_state['hotel_reviews'])
            with st.expander("Show All Similar Reviews"):
                st.text(st.session_state['all_similar_reviews'])

        st.write("\n")
        st.write("---------")
        st.write("\n")
        st.write("### About")
        st.write("The recommender uses the Hypothetical Document Embeddings (HyDE)" +
                 " approach which uses an LLM (OpenAI in this case) to generate a fake review" +
                 " based on user input. The system then uses Redis vector search to semantically search"
                 " for hotels with reviews that are similar to the fake review. The returned reviews" +
                 " are then passed to another LLM to generate a recommendation for the user.")

        st.write("#### Dataset")
        st.write("The dataset is from [Datafiniti](https://data.world/datafiniti/hotel-reviews) and is hosted on data.world")

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
            """
            % e.reason
        )'
'''