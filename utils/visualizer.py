import matplotlib

matplotlib.use('Agg')  # Fix for running on servers without a screen
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os


def generate_wordcloud(text, filename="wordcloud.png"):
    """
    Generates a word cloud image from text and saves it to static folder.
    """
    # Create static directory if it doesn't exist
    static_path = os.path.join('static')
    if not os.path.exists(static_path):
        os.makedirs(static_path)

    full_path = os.path.join(static_path, filename)

    # Generate Word Cloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

    # Save to file
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(full_path)
    plt.close()

    return filename