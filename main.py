import os
import sqlite3
from datetime import date

# -------------------------------
TEAM_NAME = "The fun gang"
DB_PATH = "C:/Users/ofeko/OneDrive/Desktop/empty_sqlite_db.db"
OUTPUT_FILE = "basics.txt"
DATE_STR = "7.9.2025"
DATABASE_NAME = "empty_sqlite_db"
# -------------------------------


def fetch_one_int(conn, query, params=None):
    cur = conn.execute(query, params or [])
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def fetch_all(conn, query, params=None):
    cur = conn.execute(query, params or [])
    return cur.fetchall()


def make_ascii_table(rows, headers):
    # rows: list of tuples [(col1, col2), ...]
    # headers: tuple/list of two header strings, e.g. ("bin", "N")
    if len(headers) != 2:
        raise ValueError("make_ascii_table supports exactly two columns")

    def sanitize(x):
        s = "" if x is None else str(x)
        # Keep table layout intact by removing newlines and replacing pipes
        return s.replace("\n", " ").replace("\r", " ").replace("|", "¦")

    str_rows = [(sanitize(a), sanitize(b)) for a, b in rows]

    col1_header, col2_header = headers
    col1_width = max(len(col1_header), *(len(r[0]) for r in str_rows)) if str_rows else len(col1_header)
    col2_width = max(len(col2_header), *(len(r[1]) for r in str_rows)) if str_rows else len(col2_header)

    border = f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+"
    header_line = f"| {col1_header.ljust(col1_width)} | {col2_header.ljust(col2_width)} |"

    lines = [border, header_line, border]

    def is_int_string(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    for a_str, b_str in str_rows:
        a_fmt = a_str.rjust(col1_width) if is_int_string(a_str) else a_str.ljust(col1_width)
        b_fmt = b_str.rjust(col2_width) if is_int_string(b_str) else b_str.ljust(col2_width)
        lines.append(f"| {a_fmt} | {b_fmt} |")
    lines.append(border)
    return "\n".join(lines)


def compute_stats(conn):
    # Counts
    num_users = fetch_one_int(conn, 'SELECT COUNT(*) FROM "BX-Users";')
    num_books = fetch_one_int(conn, 'SELECT COUNT(*) FROM "BX-Books";')
    num_ratings = fetch_one_int(conn, 'SELECT COUNT(*) FROM "BX-Book-Ratings_After_filtering";')

    # Histogram of ratings per user
    user_hist_rows = fetch_all(
        conn,
        '''
        WITH per_user AS (
            SELECT "User-ID" AS uid, COUNT(*) AS c
            FROM "BX-Book-Ratings_After_filtering"
            GROUP BY "User-ID"
        )
        SELECT c AS bin, COUNT(*) AS N
        FROM per_user
        GROUP BY c
        ORDER BY c ASC;
        '''
    )

    # Histogram of ratings per book
    book_hist_rows = fetch_all(
        conn,
        '''
        WITH per_book AS (
            SELECT "ISBN" AS isbn, COUNT(*) AS c
            FROM "BX-Book-Ratings_After_filtering"
            GROUP BY "ISBN"
        )
        SELECT c AS bin, COUNT(*) AS N
        FROM per_book
        GROUP BY c
        ORDER BY c ASC;
        '''
    )

    # Top-10 rated books (by count in filtered table)
    top_books_rows = fetch_all(
        conn,
        '''
        SELECT b."Book-Title" AS name, COUNT(*) AS N
        FROM "BX-Book-Ratings_After_filtering" r
        JOIN "BX-Books" b USING ("ISBN")
        GROUP BY r."ISBN", b."Book-Title"
        ORDER BY N DESC, name ASC
        LIMIT 10;
        '''
    )

    # Top-10 active users
    top_users_rows = fetch_all(
        conn,
        '''
        SELECT r."User-ID" AS name, COUNT(*) AS N
        FROM "BX-Book-Ratings_After_filtering" r
        GROUP BY r."User-ID"
        ORDER BY N DESC, name ASC
        LIMIT 10;
        '''
    )

    return {
        "num_users": num_users,
        "num_books": num_books,
        "num_ratings": num_ratings,
        "user_hist": user_hist_rows,
        "book_hist": book_hist_rows,
        "top_books": top_books_rows,
        "top_users": top_users_rows,
    }


def build_basics_txt(team_name, db_name, stats, date_str=None):
    today = date_str if date_str else date.today().isoformat()

    user_hist_table = make_ascii_table([(row[0], row[1]) for row in stats["user_hist"]], ("bin", "N"))
    book_hist_table = make_ascii_table([(row[0], row[1]) for row in stats["book_hist"]], ("bin", "N"))

    top_books = []
    for name, n in stats["top_books"]:
        name_str = "" if name is None else str(name).replace("\n", " ").replace("\r", " ").replace("|", "¦")
        top_books.append((name_str, n))
    top_books_table = make_ascii_table(top_books, ("name", "N"))

    top_users = []
    for name, n in stats["top_users"]:
        name_str = "" if name is None else str(name).replace("\n", " ").replace("\r", " ").replace("|", "¦")
        top_users.append((name_str, n))
    top_users_table = make_ascii_table(top_users, ("name", "N"))

    lines = []
    lines.append("-------------------- BEGIN basics.txt --------------------")
    lines.append(f"# Team: {team_name}")
    lines.append(f"# Date: {today}")
    lines.append(f"# Database name   {db_name}")
    lines.append(f"3.a) how many users?      {stats['num_users']}")
    lines.append(f"3.b) how many books?      {stats['num_books']}")
    lines.append(f"3.c) how many ratings?    {stats['num_ratings']}")
    lines.append("3.d) histogram of user-ratings <table(num ratings, num users)>")
    lines.append("     (how many users have rated N times?)")
    lines.append(user_hist_table)
    lines.append("3.e) histogram of book-ratings <table(num ratings, num users)>")
    lines.append("     (how many books have been rated N times?)")
    lines.append(book_hist_table)
    lines.append("3.f) top-10 rated books?    <table(name,num ratings)>")
    lines.append(top_books_table)
    lines.append("3.g) top-10 active users?   <table(name, num ratings)>")
    lines.append(top_users_table)
    lines.append("-------------------- END basics.txt --------------------")
    return "\n".join(lines)


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    db_name = DATABASE_NAME if DATABASE_NAME else os.path.basename(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        stats = compute_stats(conn)
    finally:
        conn.close()

    content = build_basics_txt(TEAM_NAME, db_name, stats, date_str=DATE_STR)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Written {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
