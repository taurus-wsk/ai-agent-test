import os

from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

key_extract_query_keyword = "key_extract_query_keyword"
key_search_baidu = "key_search_baidu"
key_reply_user = "key_reply_user"
def node_extract_query_keyword(state: MessagesState):
    ...
def node_search_baidu(state: MessagesState):
    ...
def node_reply_user(state: MessagesState):
    ...
def output_graph_image(graph, filename) :
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_file_dir = os.path.dirname (__file__)
        output_file_path = os. path.join(output_file_dir, filename+".png")
        with open(output_file_path, "wb") as output_file:
            output_file.write(png_data)
    except Exception as e:
        print (e)
state_graph = StateGraph(MessagesState)
state_graph.add_node(key_extract_query_keyword, node_extract_query_keyword)
state_graph.add_node(key_search_baidu, node_search_baidu)
state_graph.add_node(key_reply_user, node_reply_user)

state_graph.add_edge(START, key_extract_query_keyword)
state_graph.add_edge(key_extract_query_keyword, key_search_baidu)
state_graph.add_edge(key_search_baidu, key_reply_user)
state_graph.add_edge(key_reply_user, END)
com=state_graph.compile()
output_graph_image(com,"graph")