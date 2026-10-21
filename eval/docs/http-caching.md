# HTTP caching

Browsers and proxies store responses so they do not have to download them again. The Cache-Control header tells caches how long a response stays fresh. The max-age directive sets the freshness lifetime in seconds.

When a cached response becomes stale, the client can revalidate it. An ETag is an opaque identifier for a specific version of a resource. The client sends it back in the If-None-Match header, and the server answers with 304 Not Modified when nothing changed.

The no-store directive forbids caching entirely and is used for sensitive data. The private directive allows only the browser to cache a response, while shared caches such as CDNs must skip it.
