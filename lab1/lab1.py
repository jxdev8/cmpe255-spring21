import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # Load data from data/chipotle.tsv file using Pandas library and
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # The number of observations/entries in the dataset.
        return len(self.chipo.index)
    
    def info(self) -> None:
        # print data info.
        print(self.chipo.info())
    
    def num_column(self) -> int:
        # The number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # Print the name of all the columns.
        print(self.chipo.columns)
    
    def most_ordered_item(self):
        res = self.chipo.groupby(['item_name'])\
            .sum()\
            .sort_values(by=['quantity'], ascending=False)\
            .iloc[0]
        item_name = res.name
        order_id = res.loc['order_id']
        quantity = res.loc['quantity']
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
        # How many items were ordered in total?
        return self.chipo['quantity'].sum()

    def total_sales(self) -> float:
        # 1. Create a lambda function to change all item prices to float.
        item_price = self.chipo['item_price'].apply(lambda x: x.replace('$', '').replace(',', '')).astype(float)

        # 2. Calculate total sales.
        total_sales = (self.chipo.quantity * item_price).sum()

        return total_sales
   
    def num_orders(self) -> int:
        # How many orders were made in the dataset?
        num_orders = len(self.chipo['order_id'].unique())
        return num_orders
    
    def average_sales_amount_per_order(self) -> float:
        total_sales = self.total_sales()
        num_orders = self.num_orders()
        avg_sales_per_order = total_sales / num_orders
        return round(avg_sales_per_order, 2)

    def num_different_items_sold(self) -> int:
        # How many different items are sold?
        return len(self.chipo['item_name'].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)

        # 1. convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(letter_counter, orient='index').reset_index()
        df = df.rename(columns={'index': 'item', 0: 'count'})

        # 2. sort the values from the top to the least value and slice the first 5 items
        df = df.sort_values(by=['count'], ascending=False)[:5]

        # 3. create a 'bar' plot from the DataFrame
        items = df['item']
        counts = df['count']
        plt.bar(items, counts)

        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
        plt.title('Most popular items')

        # 5. show the plot. Hint: plt.show(block=True).
        plt.show()
        
    def scatter_plot_num_items_per_order_price(self) -> None:

        # 1. create a list of prices by removing dollar sign and trailing space.
        self.chipo['item_price'] = self.chipo['item_price']\
            .apply(lambda x: x.replace('$', '').replace(',', ''))\
            .astype(float)

        # 2. groupby the orders and sum it.
        df = self.chipo.groupby(by=['order_id']).sum()

        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        X = df.iloc[0:, 1]
        Y = df.iloc[0:, 0]

        # 4. set the title and labels.
        #       title: Number of items per order price
        #       x: Order Price
        #       y: Num Items
        plt.title('Number of items per order price')
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.scatter(X, Y, s=50, c='blue')
        plt.show()

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    """
    assert quantity == 159
    """
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()


if __name__ == "__main__":
    # execute only if run as a script
    test()
