from collections import Counter, defaultdict
import numpy as np
import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt

from loaders import ItemLoader
from items import Item


load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')





class FineTuning:

    DATASET_NAME: str = "McAuley-Lab/Amazon-Reviews-2023"
    dataset: any

    def __init__(self):
        # Log in to HuggingFace
        login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)

        dataset_names = [
            # "Automotive",
            # "Electronics",
            # "Office_Products",
            # "Tools_and_Home_Improvement",
            # "Cell_Phones_and_Accessories",
            # "Toys_and_Games",
            "Appliances",
            # "Musical_Instruments",
        ]
        items = []
        for dataset_name in dataset_names:
            loader = ItemLoader(dataset_name)
            items.extend(loader.load())
        print(f"A grand total of {len(items):,} items")

        # Visualize plots
        # self.visualize_distribution_of_token_counts(items=items)
        # self.visualize_distribution_of_prices(items=items)
        # self.visualize_distribution_of_categories(items=items)
        
        # Create a dict with a key of each price from $1 to $999
        # And in the value, put a list of items with that price (to nearest round number)

        slots = defaultdict(list)
        for item in items:
            slots[round(item.price)].append(item)
            
        # Create a dataset called "sample" which tries to more evenly take from the range of prices
        # And gives more weight to items from categories other than Automotive
        # Set random seed for reproducibility

        np.random.seed(42)
        random.seed(42)
        sample = []
        for i in range(1, 1000):
            slot = slots[i]
            if i>=240:
                sample.extend(slot)
            elif len(slot) <= 1200:
                sample.extend(slot)
            else:
                weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
                weights = weights / np.sum(weights)
                selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
                selected = [slot[i] for i in selected_indices]
                sample.extend(selected)

        print(f"There are {len(sample):,} items in the sample")

        # Visualize plots
        # self.visualize_distribution_of_prices(items=sample, color='darkblue')
        # self.visualize_distribution_of_categories(items=sample, color='lightgreen')
        # self.visualize_distribution_of_categories(items=sample, color='lightgreen', chart_type='circle')

        
        # # How does the price vary with the character count of the prompt?
        # sizes = [len(item.prompt) for item in sample]
        # prices = [item.price for item in sample]

        # # Create the scatter plot
        # plt.figure(figsize=(15, 8))
        # plt.scatter(sizes, prices, s=0.2, color="red")

        # # Add labels and title
        # plt.xlabel('Size')
        # plt.ylabel('Price')
        # plt.title('Is there a simple correlation?')

        # # Display the plot
        # plt.show()
        
        self.report(sample[398000])

    def report(self, item):
        prompt = item.prompt
        tokens = Item.tokenizer.encode(item.prompt)
        print(prompt)
        print(tokens[-10:])
        print(Item.tokenizer.batch_decode(tokens[-10:]))

    def visualize_distribution_of_token_counts(self, items: list=[]):
        tokens = [item.token_count for item in items]
        plt.figure(figsize=(15, 6))
        plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
        plt.xlabel('Length (tokens)')
        plt.ylabel('Count')
        plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
        plt.show()

    def visualize_distribution_of_prices(self, items: list=[], color: str='blueviolet'):
        prices = [item.price for item in items]
        plt.figure(figsize=(15, 6))
        plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, rwidth=0.7, color=color, bins=range(0, 1000, 10))
        plt.show()

    def visualize_distribution_of_categories(self, items: list=[], color: str="goldenrod", chart_type: str='bar'):
        category_counts = Counter()
        for item in items:
            category_counts[item.category]+=1

        categories = category_counts.keys()
        counts = [category_counts[category] for category in categories]

        if chart_type == 'basr':
            # Bar chart by category
            plt.figure(figsize=(15, 6))
            plt.bar(categories, counts, color=color)
            plt.title('How many in each category')
            plt.xlabel('Categories')
            plt.ylabel('Count')

            plt.xticks(rotation=30, ha='right')

            # Add value labels on top of each bar
            for i, v in enumerate(counts):
                plt.text(i, v, f"{v:,}", ha='center', va='bottom')

            # Display the chart
            plt.show()
        elif chart_type == 'circle':
            # Automotive still in the lead, but improved somewhat
            # For another perspective, let's look at a pie
            plt.figure(figsize=(12, 10))
            plt.pie(counts, labels=categories, autopct='%1.0f%%', startangle=90)

            # Add a circle at the center to create a donut chart (optional)
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.title('Categories')

            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis('equal')

            plt.show()

if __name__ == '__main__':
    fine_tuning = FineTuning()

