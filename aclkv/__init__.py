"""aclkv: Access-Aware KV-Cache Sharing for Resource-Constrained RAG Serving.

Layout (one research problem, increasingly capable ideas):

- ``acl``        ACL atoms, effective-prefix ACL (intersection) and audiences.
- ``scope``      Server-keyed (HMAC) fixed-size scope ids and the ``cache_salt``
                 wire encoding that carries several barriers per request.
- ``ordering``   Retrieval / access-aware / reuse-aware evidence ordering.
- ``policies``   B0..B5 + insecure upper bound; maps a request to scope barriers.
- ``context``    Builds token-id prompts with block-aligned segments.
- ``workload``   Synthetic multi-user RAG workload over HotpotQA with ACLs.
- ``simulator``  vLLM-faithful prefix-cache simulator with a security audit.
- ``vllm_plugin`` The vLLM general plugin that implements scoped block hashing.
- ``bench``      Closed-loop benchmark driver against a live vLLM server.
"""

__version__ = "0.1.0"
