# smart-ask-demo

一个最小可运行的"智能问数"示例:用自然语言提问,系统结合指标口径文档(RAG)生成 SQL(Text2SQL),在只读数据库上执行,并用中文回答。

> 这是一个学习与演示项目,使用的是程序自动生成的模拟数据,不是真实业务数据。

## 运行效果

以下是真实运行记录(模型:LongCat-2.0,数据为程序生成的模拟数据):

```
问> 毛利率最高的商品是什么?
[RAG 命中口径] ['毛利率:毛利润除以销售额 ...', '毛利润:销售额减去成本 ...']
[生成SQL 第1次] SELECT p.name AS 商品名, ROUND(SUM((p.price - p.cost) * o.qty) / SUM(p.price * o.qty) * 100, 2) || '%' AS 毛利率
                FROM orders o JOIN products p ON o.product_id = p.id
                GROUP BY p.name ORDER BY SUM((p.price - p.cost) * o.qty) / SUM(p.price * o.qty) DESC LIMIT 1
答> 毛利率最高的商品是笔记本支架,其毛利率为 62.03%。

问> 各城市销售额排名
[生成SQL 第1次] SELECT c.city, SUM(p.price * o.qty) AS sales ... GROUP BY c.city ORDER BY sales DESC
答> 1. 青岛 94,804  2. 上海 72,634  3. 北京 72,105  4. 济南 61,980  5. 潍坊 12,854  6. 烟台 9,405
```

上述结果已与独立编写的 SQL 核对,数值一致。

## 工作流程

```
用户提问
  → RAG 检索指标口径文档(如"毛利率""活跃客户"怎么算)
  → 表结构 + 命中的口径 + 示例 交给大模型,生成 SQL
  → SQL 安全校验(单条 SELECT、关键字拦截)
  → 只读连接执行(报错则把错误信息回传给模型重试,最多 2 次)
  → 大模型把查询结果总结成自然语言
```

## 设计要点

- **Text2SQL**:提示词中提供表结构(含中文字段注释)、业务口径和 few-shot 示例,温度设为 0 保证输出稳定。
- **RAG**:模型并不知道"毛利率""活跃客户"在业务里的定义,所以先检索口径文档再生成 SQL。当前检索用字符 bigram + TF-IDF + 余弦相似度,不依赖额外的 embedding 接口;可替换为 embedding + Chroma/FAISS,检索接口 `Retriever.search()` 不需要改。
- **安全(两层防护)**:
  1. 应用层校验:只允许单条 `SELECT` / `WITH` 查询,拦截 `DROP`、`UPDATE`、`DELETE`、`PRAGMA` 等关键字。
  2. 数据库层:以 `mode=ro` 只读方式打开 SQLite,即使校验被绕过也无法写入。
  另外查询结果统一限制最多返回 100 行。
- **自纠错**:SQL 执行失败时,把报错信息和上一次的 SQL 一起发给模型,让其修正后重试。
- **问口径不查数**:如果问题只是询问指标定义(如"毛利率怎么算"),模型返回 `NO_SQL`,直接基于检索到的口径回答。

## 快速开始

```bash
pip install -r requirements.txt

# 任何 OpenAI 兼容接口均可(DeepSeek / 通义 / 智谱 / OpenAI 等)
export LLM_API_KEY=你的key
export LLM_BASE_URL=https://api.deepseek.com   # 默认值
export LLM_MODEL=deepseek-chat                 # 默认值

python smart_ask.py --selftest   # 自检:不调用大模型,验证数据库、检索、安全校验、重试流程
python smart_ask.py              # 交互问答
```

Windows PowerShell 设置环境变量:`$env:LLM_API_KEY="你的key"`

示例问题:
- 毛利率最高的商品是什么
- 最近30天有多少活跃客户
- 各城市销售额排名
- 毛利率怎么计算

## 数据与口径

程序启动时自动生成 `demo.db`(SQLite):`customers`(30 个客户)、`products`(6 个商品)、`orders`(300 条订单)。口径文档在 `smart_ask.py` 的 `KB` 列表中,可自行增删。

## 已知局限

- 检索基于字符 n-gram,语义泛化能力弱于 embedding 检索。
- 单轮问答,不支持多轮追问与上下文指代。
- 复杂问题(多层嵌套、窗口函数)生成的 SQL 可能出错,目前靠报错重试兜底,无语义层面的结果校验。
- 示例库只有 3 张表;表很多时需要先做表/字段检索再拼提示词。

## 后续计划

- [ ] 检索升级为 embedding + 向量库
- [ ] 增加多轮对话
- [ ] 增加评测集,统计 SQL 生成准确率
