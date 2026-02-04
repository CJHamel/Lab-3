from typing import *
from dataclasses import dataclass
import unittest
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
sys.setrecursionlimit(10**9)

# Linked list data structure
@dataclass
class Node:
    """A node in a linked list."""
    value: int
    next: Optional['Node'] = None

# Type alias for linked list
LinkedList = Optional[Node]

# This is for reference; you can get rid of this function if you want.
def example_graph_creation() -> None:
    # Return log-base-2 of 'x' + 5.
    def f_to_graph(x: float) -> float:
        return math.log2(x) + 5.0
    
    # here we're using "list comprehensions": more of Python's
    # syntax sugar.
    x_coords: List[float] = [float(i) for i in range(1, 100)]
    y_coords: List[float] = [f_to_graph(x) for x in x_coords]
    # Could have just used this type from the start, but I want
    # to emphasize that 'matplotlib' uses 'numpy''s specific array
    # type, which is different from the built-in Python array
    # type.
    x_numpy: np.ndarray = np.array(x_coords)
    y_numpy: np.ndarray = np.array(y_coords)
    plt.plot(x_numpy, y_numpy, label='log_2(x)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Example Graph")
    plt.grid(True)
    plt.legend()  # makes the 'label's show up
    plt.show()


def range_ll(max_exclusive: int) -> LinkedList:
    """
    Accepts an integer max_exclusive larger than zero and returns a linked list 
    of the numbers from 0 up to max_exclusive - 1.
    
    Args:
        max_exclusive: An integer larger than zero
        
    Returns:
        A linked list containing integers from 0 to max_exclusive - 1
        
    Example:
        range_ll(3) returns Node(0, Node(1, Node(2, None)))
    """
    if max_exclusive <= 0:
        return None
    
    # Build the list backwards, then reverse it
    # Start with the last node
    head: LinkedList = None
    
    # Build from max_exclusive-1 down to 0
    for i in range(max_exclusive - 1, -1, -1):
        head = Node(i, head)
    
    return head


def occurs(value: int, lst: LinkedList) -> bool:
    """
    Accepts an integer and a linked list and returns whether the integer 
    occurs in the linked list.
    
    Args:
        value: The integer to search for
        lst: A linked list to search in
        
    Returns:
        True if value occurs in lst, False otherwise
        
    Example:
        occurs(2, Node(0, Node(1, Node(2, None)))) returns True
        occurs(5, Node(0, Node(1, Node(2, None)))) returns False
    """
    current = lst
    while current is not None:
        if current.value == value:
            return True
        current = current.next
    return False


class Tests(unittest.TestCase):
    def test_range_ll(self):
        # Test with max_exclusive = 1
        result = range_ll(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.value, 0)
        self.assertIsNone(result.next)
        
        # Test with max_exclusive = 3
        result = range_ll(3)
        self.assertIsNotNone(result)
        self.assertEqual(result.value, 0)
        self.assertEqual(result.next.value, 1)
        self.assertEqual(result.next.next.value, 2)
        self.assertIsNone(result.next.next.next)
        
        # Test with max_exclusive = 5
        result = range_ll(5)
        values = []
        current = result
        while current is not None:
            values.append(current.value)
            current = current.next
        self.assertEqual(values, [0, 1, 2, 3, 4])
    
    def test_occurs(self):
        # Create a linked list: 0 -> 1 -> 2 -> None
        lst = Node(0, Node(1, Node(2, None)))
        
        # Test value that exists
        self.assertTrue(occurs(0, lst))
        self.assertTrue(occurs(1, lst))
        self.assertTrue(occurs(2, lst))
        
        # Test value that doesn't exist
        self.assertFalse(occurs(5, lst))
        self.assertFalse(occurs(-1, lst))
        
        # Test empty list
        self.assertFalse(occurs(0, None))


if __name__ == '__main__':
    unittest.main()