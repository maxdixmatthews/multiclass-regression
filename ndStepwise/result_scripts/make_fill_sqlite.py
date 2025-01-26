import sqlite3
import itertools
from itertools import chain, combinations

# Function to generate all partitions of a set into two non-empty subsets
def all_partitions(elements):
    s = list(elements)
    for size in range(1, len(s)):
        for combo in combinations(s, size):
            yield tuple(combo), tuple(x for x in s if x not in combo)

# Recursive function to generate nested dichotomies
def nested_dichotomies(elements):
    if len(elements) <= 1:
        yield ()
        return

    for part1, part2 in all_partitions(elements):
        for sub1 in nested_dichotomies(part1):
            for sub2 in nested_dichotomies(part2):
                yield ((part1, part2),) + sub1 + sub2

# SQLite setup
def setup_database():
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nd_model_registry (
        name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        kfold INTEGER,
        nd_structure TEXT NOT NULL,
        model_structure TEXT NOT NULL
        accuracy NUMERIC NOT NULL,
        run_time_seconds NUMERIC,
        inner_kfolds INTEGER NOT NULL,
        run_timestamp TEXT,
        notes TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS competitor_model_registry (
        name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        kfold INTEGER,
        accuracy NUMERIC NOT NULL,
        run_time_seconds NUMERIC,
        run_timestamp TEXT,
        notes TEXT
    )
    """)
    
    conn.commit()
    return conn, cursor

# Insert data into SQLite database
def insert_dichotomies(cursor, dichotomies, batch_number):
    batch_data = [
        (str(dichotomy), batch_number) for dichotomy in dichotomies
    ]
    cursor.executemany("""
        INSERT INTO nested_dichotomies_ten (dichotomy, batch_number)
        VALUES (?, ?)
    """, batch_data)

# Main execution
if __name__ == "__main__":
    elements = tuple(range(0, 10))  # Elements for n=10
    batch_size = 10000  # Adjust this based on memory constraints

    # Setup database
    conn, cursor = setup_database()

    # Generate and insert dichotomies in batches
    batch_number = 1
    batch = []

    for i, dichotomy in enumerate(nested_dichotomies(elements), start=1):
        batch.append(dichotomy)
        
        if len(batch) >= batch_size:
            insert_dichotomies(cursor, batch, batch_number)
            conn.commit()
            print(f"Inserted batch {batch_number} with {len(batch)} dichotomies.")
            batch_number += 1
            batch = []

    # Insert any remaining dichotomies
    if batch:
        insert_dichotomies(cursor, batch, batch_number)
        conn.commit()
        print(f"Inserted final batch {batch_number} with {len(batch)} dichotomies.")

    # Close database connection
    conn.close()
