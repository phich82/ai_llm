import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt

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

        # Load in our dataset
        self.dataset = load_dataset(self.DATASET_NAME, "raw_meta_Appliances", split="full", trust_remote_code=True)
        print(f"Number of Appliances: {len(self.dataset):,}")
        # Investigate a particular datapoint
        datapoint = self.dataset[2]

        # Print values of keys in datapoint
        for key in datapoint.keys():
            print(f'Key: {key} ===> {datapoint[key]}')

        # print("1 ===> How many have prices?")
        # # How many have prices?
        # prices = 0
        # for datapoint in self.dataset:
        #     try:
        #         price = float(datapoint["price"])
        #         if price > 0:
        #             prices += 1
        #     except ValueError as e:
        #         pass
        # print(f"There are {prices:,} with prices which is {prices/len(self.dataset)*100:,.1f}%")

        # print("2 ===> So what is this item with price > 21000?")
        # # So what is this item?
        # for datapoint in self.dataset:
        #     try:
        #         price = float(datapoint["price"])
        #         if price > 21000:
        #             print(f"It is `{datapoint['title']}`")
        #     except ValueError as e:
        #         pass

        print("3 ===> Create an Item object for each with a price")
        # Create an Item object for each with a price
        items = []
        count = 1
        for datapoint in self.dataset:
            try:
                if count == 6:
                    break
                price = float(datapoint["price"])
                if price > 0:
                    print(datapoint)
                    item = Item(datapoint, price)
                    print(item.details)
                    if item.include:
                        items.append(item)
                count += 1
            except ValueError as e:
                pass

        # print(f"There are {len(items):,} items")
        # print(items[1])
        # # Investigate the prompt that will be used during training - the model learns to complete this
        # print(items[100].prompt)
        # # Investigate the prompt that will be used during testing - the model has to complete this
        # print(items[100].test_prompt())

        # # Plot the distribution of token counts
        # tokens = [item.token_count for item in items]
        # plt.figure(figsize=(15, 6))
        # plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
        # plt.xlabel('Length (tokens)')
        # plt.ylabel('Count')
        # plt.hist(tokens, rwidth=0.7, color="green", bins=range(0, 300, 10))
        # plt.show()

        # # Plot the distribution of prices
        # prices = [item.price for item in items]
        # plt.figure(figsize=(15, 6))
        # plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
        # plt.xlabel('Price ($)')
        # plt.ylabel('Count')
        # plt.hist(prices, rwidth=0.7, color="purple", bins=range(0, 300, 10))
        # plt.show()

    def visualize_lengths(self):
        # For those with prices, gather the price and the length
        prices = []
        lengths = []
        for datapoint in self.dataset:
            try:
                price = float(datapoint["price"])
                if price > 0:
                    prices.append(price)
                    contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
                    lengths.append(len(contents))
            except ValueError as e:
                pass

        # Plot the distribution of lengths
        plt.figure(figsize=(15, 6))
        plt.title(f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} and highest {max(lengths):,}\n")
        plt.xlabel('Length (chars)')
        plt.ylabel('Count')
        plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
        plt.show()

    def visualize_prices(self):
        # For those with prices, gather the price and the length
        prices = []
        lengths = []
        for datapoint in self.dataset:
            try:
                price = float(datapoint["price"])
                if price > 0:
                    prices.append(price)
                    contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
                    lengths.append(len(contents))
            except ValueError as e:
                pass

        # Plot the distribution of prices
        plt.figure(figsize=(15, 6))
        plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
        plt.show()


if __name__ == '__main__':
    fine_tuning = FineTuning()

