#!/usr/bin/env python3
"""
Natural Language Database Query Interface

This script allows users to ask questions in plain English and get answers
by querying a PostgreSQL database through OpenAI's GPT models.
"""

import os
import sys
import psycopg
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "queue"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}


def get_database_schema():
    """
    Connect to the database and retrieve the schema information.
    Returns a string describing all tables and their columns.
    """
    try:
        conn = psycopg.connect(**DB_CONFIG)
        # Set connection to read-only mode
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = ON;")

        cursor = conn.cursor()

        # Get all tables and their columns
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)

        tables = cursor.fetchall()
        schema_info = "Database Schema:\n\n"

        for (table_name,) in tables:
            schema_info += f"Table: {table_name}\n"
            schema_info += "Columns:\n"

            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))

            columns = cursor.fetchall()
            for col_name, col_type, nullable in columns:
                nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                schema_info += f"  - {col_name}: {col_type} ({nullable_str})\n"

            schema_info += "\n"

        cursor.close()
        conn.close()

        return schema_info

    except psycopg.Error as e:
        print(f"Database connection error: {e}")
        sys.exit(1)


def get_two_shot_examples(schema):
    """
    Generate two domain-specific example Q&A pairs to guide SQL generation.
    These examples help GPT understand the database structure better (two-shot learning).
    For the TA queue system, we provide realistic examples showing joins and aggregations.
    """
    # Check if this is the TA queue database by looking for key tables
    has_student_request = "student_request" in schema.lower()
    has_ta = "ta" in schema.lower()
    has_person = "person" in schema.lower()

    examples = "\nEXAMPLES (for reference on how to query this database):\n"

    if has_student_request and has_ta and has_person:
        # This is the TA queue system - provide domain-specific examples
        examples += "\nExample 1:"
        examples += "\nQuestion: How many students are waiting for help?"
        examples += "\nSQL: SELECT COUNT(*) as waiting_count FROM student_request sr WHERE NOT EXISTS (SELECT 1 FROM ta_response tr WHERE tr.student_request = sr.id);"

        examples += "\nExample 2:"
        examples += "\nQuestion: Show me the names of active TAs"
        examples += "\nSQL: SELECT p.first_name, p.last_name FROM person p JOIN ta t ON p.byu_id = t.byu_id WHERE t.active_status = TRUE;"
    else:
        # Generic fallback for unknown databases
        tables = []
        for line in schema.split("\n"):
            if line.startswith("Table:"):
                tables.append(line.replace("Table: ", "").strip())

        if tables:
            examples += "\nExample 1:"
            examples += f"\nQuestion: Show all records from {tables[0]}"
            examples += f"\nSQL: SELECT * FROM {tables[0]} LIMIT 10;"

            if len(tables) > 1:
                examples += "\nExample 2:"
                examples += f"\nQuestion: Count records in {tables[0]}"
                examples += f"\nSQL: SELECT COUNT(*) as total FROM {tables[0]};"
            else:
                examples += "\nExample 2:"
                examples += f"\nQuestion: How many records are in {tables[0]}?"
                examples += f"\nSQL: SELECT COUNT(*) as record_count FROM {tables[0]};"

    return examples


def generate_sql(question, schema, use_two_shot=False):
    """
    Use OpenAI's GPT to generate SQL based on the user's natural language question.
    If use_two_shot is True, includes domain-specific examples to guide the model.
    """
    examples_section = ""
    if use_two_shot:
        examples_section = get_two_shot_examples(schema)

    prompt = f"""Given the following database schema, generate a Postgres SQL query to answer the user's question.

{schema}{examples_section}

IMPORTANT SECURITY RULES:
- ONLY generate SELECT queries
- NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, or any data-modifying statements
- Return ONLY the Postgres SQL query, with no additional text or markdown formatting.

User's Question: {question}

SQL Query:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    sql_query = response.choices[0].message.content.strip()
    return sql_query


def clean_sql_query(sql_query):
    """
    Clean SQL query by removing markdown formatting and extra whitespace.
    Handles code blocks like ```sql ... ``` and returns clean SQL.
    """
    query = sql_query.strip()

    # Remove markdown code blocks (```sql ... ``` or ``` ... ```)
    if query.startswith("```"):
        # Remove opening ```sql or ```
        query = query.lstrip("`")
        # Remove language identifier if present (e.g., 'sql')
        if query.startswith("sql"):
            query = query[3:]
        query = query.lstrip()
        # Remove closing ```
        query = query.rstrip("`").strip()

    return query


def validate_query(sql_query):
    """
    Validate that the query is a SELECT query to prevent data modification.
    Returns True if valid (SELECT query), False otherwise.
    """
    cleaned_query = clean_sql_query(sql_query)
    query_upper = cleaned_query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return False
    return True


def execute_query(sql_query):
    """
    Execute a SQL query against the database and return the results.
    Only SELECT queries are allowed.
    """
    # Clean the SQL query (remove markdown formatting)
    cleaned_sql = clean_sql_query(sql_query)

    # Validate that this is a SELECT query
    if not validate_query(cleaned_sql):
        print(f"Error: Only SELECT queries are allowed. Your query: {cleaned_sql}")
        return None, None

    try:
        conn = psycopg.connect(**DB_CONFIG)
        # Set connection to read-only mode
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = ON;")

        cursor = conn.cursor()

        cursor.execute(cleaned_sql)

        # Check if it's a SELECT query (returns results)
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return columns, results
        else:
            cursor.close()
            conn.close()
            return None, None

    except psycopg.Error as e:
        print(f"Query execution error: {e}")
        return None, None


def format_results(columns, results):
    """
    Format query results into a readable string.
    """
    if not results:
        return "No results found."

    formatted = ""
    for row in results:
        formatted += "\n"
        for col, val in zip(columns, row):
            formatted += f"{col}: {val}\n"

    return formatted


def generate_answer(question, results_str):
    """
    Use OpenAI's GPT to generate a natural language answer based on the query results.
    """
    prompt = f"""Based on the following database query results, provide a clear and concise natural language answer to the user's question.

User's Question: {question}

Database Results:
{results_str}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    answer = response.choices[0].message.content.strip()
    return answer


def process_question(question, use_two_shot=False):
    """
    Main function to process a user's question through the entire pipeline.
    If use_two_shot is True, uses two-shot prompting for better SQL generation.
    """
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    if use_two_shot:
        print("[Mode: Two-Shot Learning]")
    print("=" * 60)

    # Step 1: Get database schema
    print("\n[1/4] Retrieving database schema...")
    schema = get_database_schema()

    # Step 2: Generate SQL from question
    print("[2/4] Generating SQL query from your question...")
    sql_query = generate_sql(question, schema, use_two_shot=use_two_shot)
    cleaned_sql = clean_sql_query(sql_query)
    print(f"Generated SQL: {cleaned_sql}\n")

    # Step 3: Execute query
    print("[3/4] Executing query...")
    columns, results = execute_query(sql_query)

    if columns is None:
        print("Error: Could not execute the query. Only SELECT queries are allowed.")
        return

    results_str = format_results(columns, results)
    print(f"Query Results: {results_str}")

    # Step 4: Generate natural language answer
    print("[4/4] Generating natural language answer...")
    answer = generate_answer(question, results_str)

    print("\n" + "-" * 60)
    print(f"Answer: {answer}")
    print("-" * 60)


def main():
    """
    Main entry point for the application.
    Supports zero-shot (default) and two-shot prompting modes.
    """
    print("Natural Language Database Query Interface")
    print("=" * 60)
    print("Ask questions about your database in plain English!\n")

    # Check for environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY not found. Please set it in your .env file."
        )
        sys.exit(1)

    if not os.getenv("PG_USER") or not os.getenv("PG_PASSWORD"):
        print("Error: Database credentials not found. Please set them in your .env file.")
        sys.exit(1)

    # Check for two-shot flag
    use_two_shot = "--two-shot" in sys.argv or "-2" in sys.argv

    # Remove flags from argv for question processing
    filtered_argv = [arg for arg in sys.argv[1:] if arg not in ["--two-shot", "-2"]]

    if filtered_argv:
        # Process command-line argument as a question
        question = " ".join(filtered_argv)
        process_question(question, use_two_shot=use_two_shot)
    else:
        # Interactive mode
        if use_two_shot:
            print("[Two-Shot Mode Enabled]")
        print("Type 'exit' or 'quit' to exit.\n")
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if question.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not question:
                    print("Please enter a valid question.")
                    continue
                process_question(question, use_two_shot=use_two_shot)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
