# app.py
import streamlit as st
import numpy as np
import os
import keras

# ----------------- SETUP -----------------
# Set Hugging Face token securely via Streamlit secrets
# Make sure you added HF_TOKEN in Streamlit Cloud secrets
os.environ["HF_HUB_TOKEN"] = st.secrets["HF_TOKEN"]

# Force Keras to use TensorFlow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load model from Hugging Face
@st.cache_resource
def load_model():
    return keras.saving.load_model("hf://beejaytmg/ai_tic_tac_toe")

model = load_model()

# ----------------- GAME STATE -----------------
if "board" not in st.session_state:
    st.session_state.board = [0] * 9  # 0=empty,1=player,2=AI
if "winner" not in st.session_state:
    st.session_state.winner = None
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "ai_first" not in st.session_state:
    st.session_state.ai_first = False
if "game_started" not in st.session_state:
    st.session_state.game_started = False

# ----------------- GAME LOGIC -----------------
def check_winner(board):
    win_patterns = [
        [0,1,2],[3,4,5],[6,7,8],  # rows
        [0,3,6],[1,4,7],[2,5,8],  # columns
        [0,4,8],[2,4,6]           # diagonals
    ]
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if all(cell != 0 for cell in board):
        return 0  # draw
    return None  # no winner yet

def ai_move():
    prediction = model.predict(np.array([st.session_state.board]))
    move = int(np.argmax(prediction))
    if st.session_state.board[move] == 0:
        st.session_state.board[move] = 2
    else:
        for i in range(9):
            if st.session_state.board[i] == 0:
                st.session_state.board[i] = 2
                break

def handle_click(index):
    if st.session_state.board[index] == 0 and not st.session_state.game_over:
        st.session_state.board[index] = 1  # player move
        winner = check_winner(st.session_state.board)
        if winner is not None:
            st.session_state.winner = winner
            st.session_state.game_over = True
            return
        ai_move()
        winner = check_winner(st.session_state.board)
        if winner is not None:
            st.session_state.winner = winner
            st.session_state.game_over = True

def restart_game():
    st.session_state.board = [0] * 9
    st.session_state.winner = None
    st.session_state.game_over = False
    st.session_state.game_started = False

# ----------------- UI -----------------
st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🤖", layout="centered")
st.title("🤖 Tic-Tac-Toe AI by Bijay")
st.write("Play against")

# Choose who goes first
if not st.session_state.game_started:
    st.session_state.ai_first = st.radio(
        "Who plays first?",
        ["You", "AI"],
        horizontal=True
    ) == "AI"

    if st.button("Start Game"):
        st.session_state.game_started = True
        if st.session_state.ai_first:
            ai_move()

# Render board only after game started
if st.session_state.game_started:
    st.markdown("---")
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            cell = st.session_state.board[idx]
            label = " " if cell == 0 else ("🔵" if cell == 1 else "🔴")
            with cols[j]:
                st.button(
                    label,
                    key=f"cell_{idx}",
                    on_click=handle_click,
                    args=(idx,),
                    use_container_width=True
                )

    # Show result
    if st.session_state.winner is not None:
        if st.session_state.winner == 0:
            st.info("😐 It's a draw!")
        elif st.session_state.winner == 1:
            st.success("🎉 You win!")
        elif st.session_state.winner == 2:
            st.error("🤖 AI wins!")
        st.button("🔄 Restart Game", on_click=restart_game)
