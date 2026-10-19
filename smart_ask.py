"""智能问数 Demo:Text2SQL + RAG。用法见 README.md。"""
import os, re, sys, math, random, sqlite3
try:
    import readline  # fix CJK backspace in input()
except ImportError:
    pass
from collections import Counter

DB_PATH = "demo.db"

# ----------------------------------------------------------------- 1. 数据库
SCHEMA = """\
customers(id INTEGER 客户ID, name TEXT 客户名, city TEXT 城市)
products(id INTEGER 商品ID, name TEXT 商品名, category TEXT 品类, price REAL 单价, cost REAL 单位成本)
orders(id INTEGER 订单ID, customer_id INTEGER 客户ID, product_id INTEGER 商品ID, qty INTEGER 数量, order_date TEXT 下单日期YYYY-MM-DD)
关系: orders.customer_id=customers.id, orders.product_id=products.id"""


def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    random.seed(42)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT);
    CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, cost REAL);
    CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, qty INTEGER, order_date TEXT);
    """)
    cities = ["济南", "青岛", "烟台", "潍坊", "北京", "上海"]
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(i, f"客户{i:02d}", random.choice(cities)) for i in range(1, 31)])
    prods = [("机械键盘", "外设", 299, 180), ("无线鼠标", "外设", 99, 45), ("显示器", "显示", 1299, 950),
             ("笔记本支架", "配件", 79, 30), ("USB集线器", "配件", 59, 35), ("耳机", "音频", 399, 220)]
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [(i + 1, *p) for i, p in enumerate(prods)])
    rows = []
    for oid in range(1, 301):
        d = f"2026-{random.randint(7, 10):02d}-{random.randint(1, 28):02d}"
        rows.append((oid, random.randint(1, 30), random.randint(1, 6), random.randint(1, 5), d))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()


# ----------------------------------------------------------------- 2. RAG:指标口径知识库
KB = [
    "销售额:订单金额之和,销售额 = SUM(products.price * orders.qty)。",
    "毛利润:销售额减去成本,毛利润 = SUM((products.price - products.cost) * orders.qty)。",
    "毛利率:毛利润除以销售额,毛利率 = SUM((price-cost)*qty) / SUM(price*qty),结果用百分比表示。",
    "活跃客户:最近30天内至少下过1单的客户。'今天'以 orders 表中最大的 order_date 为准,而不是系统当前时间。",
    "客单价:销售额除以下单的客户数(按客户去重),客单价 = SUM(price*qty) / COUNT(DISTINCT customer_id)。",
    "复购客户:下单次数大于等于2次的客户。",
]


def tokens(text):
    s = re.sub(r"\s+", "", text.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)]  # 字符 bigram,对中文足够用


class Retriever:
    """极简向量检索:字符bigram的 TF-IDF + 余弦相似度。
    想升级成真正的 embedding 检索:把 vec() 换成调用 embedding 接口,存进 Chroma/FAISS 即可,其余逻辑不变。"""

    def __init__(self, docs):
        self.docs = docs
        toks = [tokens(d) for d in docs]
        df = Counter(t for ts in toks for t in set(ts))
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vecs = [self.vec(ts) for ts in toks]

    def vec(self, ts):
        c = Counter(ts)
        v = {t: f * self.idf.get(t, 1.0) for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=2, min_score=0.05):
        q = self.vec(tokens(query))
        scored = [(sum(w * d.get(t, 0) for t, w in q.items()), doc) for d, doc in zip(self.vecs, self.docs)]
        scored.sort(reverse=True)
        return [doc for s, doc in scored[:k] if s >= min_score]


# ----------------------------------------------------------------- 3. 大模型调用
def llm(messages):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    r = client.chat.completions.create(model=os.environ.get("LLM_MODEL", "deepseek-chat"),
                                       messages=messages, temperature=0)
    return r.choices[0].message.content.strip()


# ----------------------------------------------------------------- 4. Text2SQL
FEW_SHOT = """\
问题: 各城市的客户数量
SQL: SELECT city, COUNT(*) AS cnt FROM customers GROUP BY city ORDER BY cnt DESC
问题: 销售额最高的品类
SQL: SELECT p.category, SUM(p.price*o.qty) AS sales FROM orders o JOIN products p ON o.product_id=p.id GROUP BY p.category ORDER BY sales DESC LIMIT 1"""


def build_prompt(question, contexts, error=None, last_sql=None):
    ctx = "\n".join(f"- {c}" for c in contexts) or "(无)"
    p = f"""你是数据分析助手,把用户问题转成 SQLite 的 SELECT 语句。
【表结构】
{SCHEMA}
【业务口径】(必须遵守)
{ctx}
【示例】
{FEW_SHOT}
规则:只输出一条 SELECT 语句,不要解释;如果问题只是询问指标定义、不需要查数据,只输出 NO_SQL。
问题: {question}"""
    if error:
        p += f"\n\n上一次生成的SQL:\n{last_sql}\n执行报错:{error}\n请修正后重新输出SQL。"
    return p


def extract_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|vacuum)\b", re.I)


def check_sql(sql):
    """安全校验:只允许单条只读查询。"""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("只允许单条语句")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("只允许 SELECT 查询")
    if FORBIDDEN.search(s):
        raise ValueError("包含被禁止的关键字")
    return s


def run_sql(sql, limit=100):
    s = check_sql(sql)
    # 第二道保险:数据库连接本身就是只读的
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(f"SELECT * FROM ({s}) LIMIT {limit}")
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()
    finally:
        con.close()


def ask(question, retriever, max_retry=2, verbose=True):
    contexts = retriever.search(question)
    if verbose:
        print("[RAG 命中口径]", contexts)
    error = last_sql = None
    for attempt in range(max_retry + 1):
        out = llm([{"role": "user", "content": build_prompt(question, contexts, error, last_sql)}])
        if out.strip().upper().startswith("NO_SQL"):
            return llm([{"role": "user", "content": f"根据下面的业务口径回答问题。\n口径:{contexts}\n问题:{question}"}])
        last_sql = extract_sql(out)
        if verbose:
            print(f"[生成SQL 第{attempt + 1}次]", last_sql)
        try:
            cols, rows = run_sql(last_sql)
            break
        except Exception as e:  # 报错 -> 把错误回传给模型重试
            error = str(e)
            if verbose:
                print("[执行失败]", error)
    else:
        return f"抱歉,多次尝试后仍无法生成可执行的查询。最后的错误:{error}"
    table = "\n".join([" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows[:20]])
    return llm([{"role": "user", "content": f"用户问题:{question}\n查询结果:\n{table}\n请用简洁的中文回答,数字保留合理小数。"}])


# ----------------------------------------------------------------- 5. 入口
def selftest():
    """不调用大模型,只验证数据库 / 安全校验 / 检索 / 重试流程。"""
    global llm
    init_db()
    r = Retriever(KB)
    assert any("毛利率" in d for d in r.search("毛利率最高的商品是什么")), "RAG 检索失败"
    assert any("活跃客户" in d for d in r.search("最近有多少活跃客户")), "RAG 检索失败"
    for bad in ["DROP TABLE orders", "SELECT 1; DELETE FROM orders", "UPDATE orders SET qty=0", "PRAGMA table_info(orders)"]:
        try:
            run_sql(bad)
            raise SystemExit(f"安全校验漏洞: {bad}")
        except ValueError:
            pass
    calls = []

    def fake(messages):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return "```sql\nSELECT nope FROM orders\n```"  # 故意写错,测试重试
        if len(calls) == 2:
            return "SELECT p.name, ROUND(SUM((p.price-p.cost)*o.qty)*1.0/SUM(p.price*o.qty),3) AS gm FROM orders o JOIN products p ON o.product_id=p.id GROUP BY p.name ORDER BY gm DESC LIMIT 1"
        return "毛利率最高的是X"

    llm = fake
    print(ask("毛利率最高的商品是什么", r))
    assert "执行报错" in calls[1], "重试时没有带上错误信息"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    init_db()
    ret = Retriever(KB)
    print("智能问数 Demo 已就绪,示例:毛利率最高的商品是什么?/ 最近30天有多少活跃客户?/ 各城市销售额排名")
    while True:
        q = input("\n问> ").strip()
        if q in ("", "q", "quit", "exit"):
            break
        print("答>", ask(q, ret))
