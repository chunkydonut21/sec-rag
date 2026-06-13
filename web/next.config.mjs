/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emit a self-contained server for a small production Docker image.
  output: "standalone",
};

export default nextConfig;
