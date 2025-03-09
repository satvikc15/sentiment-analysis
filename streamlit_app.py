import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import string

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)  # For sentiment analysis
nltk.download('punkt', quiet=True, force=True)  # Base tokenizer resource
nltk.download('punkt_tab', quiet=True)  # Additional tokenizer resource
nltk.download('stopwords', quiet=True)  # For filtering common words

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Set up the Streamlit app title and description
st.title("Sentiment Analysis Dashboard")
st.write("Upload an Excel file with a 'text' column containing your comments.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        if 'text' not in df.columns:
            st.error("The uploaded Excel file does not contain a 'text' column.")
        else:
            # Extract comments and drop any missing values
            comments = df['text'].dropna().tolist()
            st.write(f"Analyzing {len(comments)} comments...")

            # Analyze sentiment for each comment
            analysis_results = []
            tones = []
            for comment in comments:
                scores = sid.polarity_scores(comment)
                compound = scores['compound']
                # Classify the comment based on the compound score
                if compound >= 0.05:
                    tone = "Positive"
                elif compound <= -0.05:
                    tone = "Negative"
                else:
                    tone = "Neutral"
                tones.append(tone)
                analysis_results.append({"comment": comment, "tone": tone, "scores": scores})

            # Create a DataFrame from the analysis results
            results_df = pd.DataFrame(analysis_results)  # Fixed missing line

            # Count the number of comments by tone
            tone_counts = results_df['tone'].value_counts()
            st.subheader("Tone Distribution (Line Graph)")
            # Prepare data for the line graph
            tones_order = ['Positive', 'Neutral', 'Negative']
            counts = [tone_counts.get(t, 0) for t in tones_order]

            # Plotting using matplotlib
            fig, ax = plt.subplots()
            ax.plot(tones_order, counts, marker='o', linestyle='-', color='blue')
            ax.set_xlabel("Tone")
            ax.set_ylabel("Number of Comments")
            ax.set_title("Number of Comments by Tone")
            st.pyplot(fig)

            # Analyze frequent words across all comments
            st.subheader("Frequent Words in Comments")
            all_text = " ".join(comments)
            tokens = word_tokenize(all_text.lower())
            # Remove punctuation and stopwords
            tokens = [word for word in tokens if word not in string.punctuation]
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            word_freq = Counter(tokens)
            top_words = word_freq.most_common(5)  # Get the top 5 words
            st.write("Top frequent words:", top_words)

            # Display comments that contain these top frequent words
            st.write("Comments containing the top frequent words:")
            for word, freq in top_words:
                with st.expander(f"Word: {word} (Frequency: {freq})"):
                    filtered_comments = [comment for comment in comments if word in comment.lower()]
                    for comment in filtered_comments:
                        st.write(f"- {comment}")

            # Provide insights from negative comments for improvement
            st.subheader("Areas for Improvement from Negative Comments")
            negative_comments = [comment for comment, tone in zip(comments, tones) if tone == "Negative"]
            if negative_comments:
                neg_text = " ".join(negative_comments)
                neg_tokens = word_tokenize(neg_text.lower())
                neg_tokens = [word for word in neg_tokens if word not in string.punctuation]
                neg_tokens = [word for word in neg_tokens if word not in stop_words]
                neg_word_freq = Counter(neg_tokens)
                neg_top_words = neg_word_freq.most_common(5)
                st.write("Key words in negative comments (potential areas to improve):", neg_top_words)
                st.write("Based on these keywords, consider investigating recurring issues that might be driving the negative feedback.")
            else:
                st.write("There are no negative comments to analyze for improvement.")

    except Exception as e:
        st.error(f"Error processing file: {e}")