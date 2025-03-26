
import os
import openai
import pickle
import streamlit as st
import torch
import pandas as pd

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
    get_user_article_titles,
    search_similar_articles,
    get_article_body_by_id,
    get_user_by_article_id,
    get_article_title_by_id,
)

from Drone_Picker import (
    extract_restrictions_from_pdf,
    uploaded_file_to_bytes,
    search_drones,
    search_drones_by_weight,
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

def flatten_json(nested_json, parent_key='', sep='_'):
    items = []
    
    # Check if the data is a dictionary
    if isinstance(nested_json, dict):
        for k, v in nested_json.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is a dictionary, recursively flatten it
                items.extend(flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # If the value is a list, iterate through each item and flatten it
                for i, elem in enumerate(v):
                    items.extend(flatten_json({f"{i}": elem}, new_key, sep=sep).items())
            else:
                # If it's a scalar value, add it to the list
                items.append((new_key, v))
    elif isinstance(nested_json, list):
        # If the data is a list, iterate through each item and treat each as an element
        for i, elem in enumerate(nested_json):
            items.extend(flatten_json({f"{i}": elem}, parent_key, sep=sep).items())
    
    return dict(items)

def main(): 
# Article Recommender
    # Streamlit UI
    st.sidebar.title("User Panel")

    # Input for username at the top of the sidebar
    username = st.sidebar.text_input("Enter your username")
    user_id = get_user_id(username)

    # Button to toggle article submission panel
    if st.sidebar.button("➕ Add New Article"):
        state.ArticleState = state.ArticleState.ADD_ARTICLE

    if st.sidebar.button("Recommend Articles"):
        state.ArticleState = state.ArticleState.RECOMMEND_ARTICLE

    if st.sidebar.button("My Articles"):
        state.ArticleState = state.ArticleState.MY_ARTICLES

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
            state.SELECTED_ARTICLE = 0

            if not user_id:
                st.write("❌ Username not found")  
            else:    
                articles = get_user_article_titles(user_id)
                if not articles:
                    st.write("❌ Article(s) not found")
                else:
                   # Sort user's articles by article_id in descending order and select the latest 3
                    latest_articles = sorted(articles, key=lambda x: x[0], reverse=True)[:3]

                    state.MY_ARTICLE_TITLES = latest_articles

                    # Iterate over the latest 3 articles and fetch similar articles
                    for i, article in enumerate(latest_articles):
                        article_id, article_title = article  # Unpack the tuple into id and title

                        # Use the article_id in the search_similar_articles function
                        similar_articles = search_similar_articles(article_id, user_id)

                        if not hasattr(state, 'SIMILAR_ARTICLES'):
                            state.SIMILAR_ARTICLES = {}

                        # Save the entire list of similar articles in state.SIMILAR_ARTICLES
                        state.SIMILAR_ARTICLES[article_id] = similar_articles  # Storing the whole array under article_id

        if state.MY_ARTICLE_TITLES is not None:

            state.SELECTED_ARTICLE = 0

            for i, article in enumerate(state.MY_ARTICLE_TITLES):
                # Unpack the article to get its title
                article_id, article_title = article

                # Print the article title
                st.write(f"##### Article title: {article_title}")

                # Get similar articles for the current article
                similar_articles = state.SIMILAR_ARTICLES.get(article_id, [])

                if not similar_articles:    
                    st.write("⚠️ No similar articles found.")
                elif isinstance(similar_articles, list) and isinstance(similar_articles[0], dict):
                    for similar_article in similar_articles:
                        # Ensure each button has a unique key by using both article_id and similar_article['id']
                        button_key = f"{article_id}_{similar_article['id']}"

                        # Create two columns: one for the title and one for the button
                        col1, col2 = st.columns([3, 1])  # Adjust column widths (3 for title, 1 for the button)

                        with col1:
                            # Display the similar article title and the author's username
                            st.write(f"**Similar article title:** {similar_article['title']} - {get_user_by_article_id(similar_article['id'])}")

                        with col2:
                            # Create a button to read the full article with a unique key
                            if st.button(f"Read Article", key=button_key):
                                state.SELECTED_ARTICLE = similar_article['id']
                                
                        if (similar_article['id'] == state.SELECTED_ARTICLE):
                            article_body = get_article_body_by_id(similar_article['id'])
                            st.write(f"\n{article_body}")
                        
                else:
                    st.write("⚠️ Unexpected format for similar_articles.")



    if state.ArticleState == state.ArticleState.MY_ARTICLES:
        st.title("My Articles")

        articles = get_user_article_titles(user_id)  # Fetch user article titles (id, title)

        for i, article in enumerate(articles):  # Use enumerate to get index of the article
            article_id, article_title = article  # Unpack the tuple into id and title

            # Display the article title
            st.write(f"**Article title:** {article_title}")

            # Create a button for each article
            if st.button(f"Expand", key=article_id):
                # Fetch the article body using the article ID
                article_body = get_article_body_by_id(article_id)  # Fetch the body by article_id
                
                # Display the article body under the title when the button is clicked
                if article_body:
                    st.write(f"\n{article_body}")
                else:
                    st.write("⚠️ Article body not found.")

''' # Drone Picket
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 65% !important;
                margin: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
)
    

    uploaded_file = st.file_uploader("Upload drone regulations PDF", type="pdf")

    if uploaded_file:
        if st.button("Extract Restrictions"):
            restrictions_data = extract_restrictions_from_pdf(uploaded_file_to_bytes(uploaded_file))

            if restrictions_data:
                st.write("✅ **Drone Restrictions Extracted**")

                state.DRONE_RESTICTION_RESULT = restrictions_data             

    if (state.DRONE_RESTICTION_RESULT):
        restrictions_data = state.DRONE_RESTICTION_RESULT
        for category, data in restrictions_data.get("drone_restrictions", {}).items():
                    min_weight = data["minimum_weight"]
                    max_weight = data["maximum_weight"]
                    restrictions_list = data["restrictions"]

                    # Display category info
                    with st.expander(f"📌 {category} ({min_weight}g - {max_weight}g)"):
                        st.write("#### 🛑 Restrictions:")
                        flat_json = flatten_json(restrictions_list)
                        df = pd.DataFrame([flat_json])
                        df_vertical = df.transpose()
                        st.dataframe(df_vertical)

                            # Search button for this weight category
                    if st.button(f"🔎 Search Drones ({min_weight}g - {max_weight}g)", key=category):
                        state.DRONE_SEARCH_DATA = data
                        
                    if (state.DRONE_SEARCH_DATA == data):
                        st.write(f"🔍 Searching drones between {min_weight}g and {max_weight}g...")
                        
                        # Search for matching drones
                        drones = search_drones_by_weight(min_weight, max_weight)
                        
                        # Display search results
                        if drones:
                            # Convert query results into a list of dictionaries for display
                            drone_data = [
                                {
                                    "Model": drone.drone_model,
                                    "Weight (g)": drone.weight_g,
                                    "Battery (mAh)": drone.battery_mah,
                                    "Flight Time (min)": drone.flight_time_min,
                                    "Max Range (km)": drone.max_range_km,
                                    "Camera": drone.camera,
                                    "Geofencing": "Yes" if drone.geofencing else "No",
                                    "Noise (dB)": drone.noise_db,
                                }
                                for drone in drones
                            ]

                            st.table(drone_data)  # Displays data in a structured table
                        else:
                            st.write("⚠️ No drones found in this weight category.")
                       


    drone_name = st.text_input("Search Drone")

    if st.button("Search"):
        drones = search_drones(drone_model__contains=drone_name)

        if drones:
            # Convert query results into a list of dictionaries for display
            drone_data = [
                {
                    "Model": drone.drone_model,
                    "Weight (g)": drone.weight_g,
                    "Battery (mAh)": drone.battery_mah,
                    "Flight Time (min)": drone.flight_time_min,
                    "Max Range (km)": drone.max_range_km,
                    "Camera": drone.camera,
                    "Geofencing": "Yes" if drone.geofencing else "No",
                    "Noise (dB)": drone.noise_db,
                }
                for drone in drones
            ]

            st.table(drone_data)  # Displays data in a structured table
        else:
            st.warning("🚫 No drones found.")
'''

''' # Article Recommender
    # Streamlit UI
    st.sidebar.title("User Panel")

    # Input for username at the top of the sidebar
    username = st.sidebar.text_input("Enter your username")
    user_id = get_user_id(username)

    # Button to toggle article submission panel
    if st.sidebar.button("➕ Add New Article"):
        state.ArticleState = state.ArticleState.ADD_ARTICLE

    if st.sidebar.button("Recommend Articles"):
        state.ArticleState = state.ArticleState.RECOMMEND_ARTICLE

    if st.sidebar.button("My Articles"):
        state.ArticleState = state.ArticleState.MY_ARTICLES

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
            state.SELECTED_ARTICLE = 0

            if not user_id:
                st.write("❌ Username not found")  
            else:    
                articles = get_user_article_titles(user_id)
                if not articles:
                    st.write("❌ Article(s) not found")
                else:
                   # Sort user's articles by article_id in descending order and select the latest 3
                    latest_articles = sorted(articles, key=lambda x: x[0], reverse=True)[:3]

                    state.MY_ARTICLE_TITLES = latest_articles

                    # Iterate over the latest 3 articles and fetch similar articles
                    for i, article in enumerate(latest_articles):
                        article_id, article_title = article  # Unpack the tuple into id and title

                        # Use the article_id in the search_similar_articles function
                        similar_articles = search_similar_articles(article_id, user_id)

                        if not hasattr(state, 'SIMILAR_ARTICLES'):
                            state.SIMILAR_ARTICLES = {}

                        # Save the entire list of similar articles in state.SIMILAR_ARTICLES
                        state.SIMILAR_ARTICLES[article_id] = similar_articles  # Storing the whole array under article_id

        if state.MY_ARTICLE_TITLES is not None:

            state.SELECTED_ARTICLE = 0

            for i, article in enumerate(state.MY_ARTICLE_TITLES):
                # Unpack the article to get its title
                article_id, article_title = article

                # Print the article title
                st.write(f"##### Article title: {article_title}")

                # Get similar articles for the current article
                similar_articles = state.SIMILAR_ARTICLES.get(article_id, [])

                if not similar_articles:    
                    st.write("⚠️ No similar articles found.")
                elif isinstance(similar_articles, list) and isinstance(similar_articles[0], dict):
                    for similar_article in similar_articles:
                        # Ensure each button has a unique key by using both article_id and similar_article['id']
                        button_key = f"{article_id}_{similar_article['id']}"

                        # Create two columns: one for the title and one for the button
                        col1, col2 = st.columns([3, 1])  # Adjust column widths (3 for title, 1 for the button)

                        with col1:
                            # Display the similar article title and the author's username
                            st.write(f"**Similar article title:** {similar_article['title']} - {get_user_by_article_id(similar_article['id'])}")

                        with col2:
                            # Create a button to read the full article with a unique key
                            if st.button(f"Read Article", key=button_key):
                                state.SELECTED_ARTICLE = similar_article['id']
                                
                        if (similar_article['id'] == state.SELECTED_ARTICLE):
                            article_body = get_article_body_by_id(similar_article['id'])
                            st.write(f"\n{article_body}")
                        
                else:
                    st.write("⚠️ Unexpected format for similar_articles.")



    if state.ArticleState == state.ArticleState.MY_ARTICLES:
        st.title("My Articles")

        articles = get_user_article_titles(user_id)  # Fetch user article titles (id, title)

        for i, article in enumerate(articles):  # Use enumerate to get index of the article
            article_id, article_title = article  # Unpack the tuple into id and title

            # Display the article title
            st.write(f"**Article title:** {article_title}")

            # Create a button for each article
            if st.button(f"Expand", key=article_id):
                # Fetch the article body using the article ID
                article_body = get_article_body_by_id(article_id)  # Fetch the body by article_id
                
                # Display the article body under the title when the button is clicked
                if article_body:
                    st.write(f"\n{article_body}")
                else:
                    st.write("⚠️ Article body not found.")
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