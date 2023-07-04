<p align="center">
  <img src="https://github.com/0hq/tinyvector/blob/main/assets/TINYVECTORLOGO.png?raw=true" alt="tinyvector logo">
</p>


<p align="center">
    <b>tinyvector - the tiny, least-dumb, speedy vector embedding database</b>. <br />
    No, you don't need a vector database. You need tinyvector.
</p>
<p align="center">
    <i>In pre-release: prod-ready by late-July.</i> <b><i>Still in development!</i></b> <br />
</p>


## Features
- __Tiny__: It's in the name. It's just a Flask server, SQLite DB, and Numpy indexes. Extremely easy to customize, under 500 lines of code.
- __Fast__: Tinyvector wlll have comparable speed to advanced vector databases when it comes to speed on small to medium datasets.
- __Vertically Scales__: Tinyvector stores all indexes in memory for fast querying. Very easy to scale up to 100 million+ vector dimensions without issue.
- __Open Source__: MIT Licensed, free forever.

### Soon
- __Powerful Queries__: Tinyvector is being upgraded with full SQL querying functionality, something missing from most other databases.
- __Integrated Models__: Soon you won't have to bring your own vectors, just generate them on the server automaticaly. Will support SBert, Hugging Face models, OpenAI, Cohere, etc.
- __Python/JS Client__: We'll add a comprehensive Python and Javascript package for easy integration with tinyvector in the next two weeks.

## We're better than ...

In most cases, most vector databases are overkill for something simple like:
1. Using embeddings to chat with your documents. Most document search is nowhere close to what you'd need to justify accelerating search speed with [HNSW](https://github.com/nmslib/hnswlib) or [FAISS](https://github.com/facebookresearch/faiss).
2. Doing search for your website or store. Unless you're selling 1,000,000 items, you don't need Pinecone.
3. Performing complex search queries on a very large database. Even if you have 2 million embeddings, this might still be the better option due to vector databases struggling with complex filtering. Tinyvector doesn't support metadata/filtering just yet, but it's very easy for you to add that yourself.

## Embeddings?

What are embeddings?

> As simple as possible: Embeddings are a way to compare similar things, in the same way humans compare similar things, by converting text into a small list of numbers. Similar pieces of text will have similar numbers, different ones have very different numbers.

Read OpenAI's [explanation](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).  


## Get involved

tinyvector is going to be growing a lot (don't worry, will still be tiny). Feel free to make a PR and contribute. If you have questions, just mention [@willdepue](https://twitter.com/willdepue).

Some ideas for first pulls:

- Add metadata and allow querying/filtering. This is especially important since a lot vector databases literally don't have a WHERE clause lol (or just an extremely weak one). Not a problem here. [Read more about this.](https://www.pinecone.io/learn/vector-search-filtering)
- Rethinking SQLite and choosing something. NOSQL feels fitting for embeddings?
- Add embedding functions for easy adding text (sentence transformers, OpenAI, Cohere, etc.)
-  Let's start GPU accelerating with a Pytorch index. GPUs are great at matmuls -> NN search with a fused kernel. Let's put 32 million vectors on a single GPU.
- Help write unit and integration tests.
- See all [active issues](https://github.com/0hq/tinyvector/issues)!

### Known Issues
```
# Major bugs:
Data corruption SQLite error? Stored vectors end up changing. Replicate by creating a table, inserting vectors, creating an index and then screwing around till an error happens. Dims end up unmatched (might be the blob functions or the norm functions most likely, but doesn't explain why the database is changing).
PCA is not tested, neither is immutable Brute Force index.
```


## License

[MIT](./LICENSE)
