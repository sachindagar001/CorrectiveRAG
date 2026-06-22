/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    // CRAG queries take 30-60s (6+ LLM calls). The default rewrite proxy
    // timeout in the dev server is ~30s, which aborts long queries with a 500.
    // Raise it to 180s so the proxy waits long enough for the backend.
    experimental: {
        proxyTimeout: 180000,
    },
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/api/:path*'
        }
      ]
    }
}
module.exports = nextConfig
