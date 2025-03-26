from enum import Enum

class BookAppState(Enum):
    DEFAULT = 0
    IS_NOT_FOUND = 1
    IS_BOOK_FOUND = 2
    IS_BOOK_SELECTED = 3

CURRENT_STATE: BookAppState = BookAppState.DEFAULT

SELECTED_BOOK_ID: object
SELECTED_BOOKS_ID: object

class ArticleState(Enum):
    ADD_ARTICLE = 0
    RECOMMEND_ARTICLE = 1
    MY_ARTICLES = 2

CURRENT_STATE: ArticleState = ArticleState.ADD_ARTICLE
MY_ARTICLE_TITLES = []
SIMILAR_ARTICLES = {}
SELECTED_ARTICLE: object

DRONE_RESTICTION_RESULT = []
DRONE_SEARCH_DATA = []