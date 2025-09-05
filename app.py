import streamlit as st
import numpy as np
import keras
import os
from huggingface_hub import login

# ----------------- SETUP -----------------
HUGGINGFACE_TOKEN = st.secrets["HF_TOKEN"]
login(token=HUGGINGFACE_TOKEN)

os.environ["KERAS_BACKEND"] = "tensorflow"

@st.cache_resource
def load_model():
    return keras.saving.load_model("hf://beejaytmg/ai_tic_tac_toe")

model = load_model()

# ----------------- GAME STATE -----------------
if "board" not in st.session_state:
    st.session_state.board = [0] * 9
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
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for a, b, c in win_patterns:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if all(cell != 0 for cell in board):
        return 0
    return None

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
        st.session_state.board[index] = 1
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
st.set_page_config(page_title="Tic-Tac-Toe AI made by Bijay", page_icon="🤖", layout="centered")
st.title("🤖 Tic-Tac-Toe AI")
st.write("Play against an AI")

# Game start options
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

# Render board only after game starts
if st.session_state.game_started:
    st.markdown("---")
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            cell = st.session_state.board[idx]
            label = " " if cell == 0 else ("🔵" if cell == 1 else "🔴")
            # Make buttons bigger and centered
            with cols[j]:
                st.button(
                    label,
                    key=f"cell_{idx}",
                    on_click=handle_click,
                    args=(idx,),
                    use_container_width=True,
                    help="Click to place your move"
                )

    if st.session_state.winner is not None:
        if st.session_state.winner == 0:
            st.info("😐 It's a draw!")
        elif st.session_state.winner == 1:
            st.success("🎉 You win!")
        elif st.session_state.winner == 2:
            st.error("🤖 AI wins!")
        st.button("🔄 Restart Game", on_click=restart_game)
