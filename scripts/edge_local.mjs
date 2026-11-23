/**
 * Run the real Cloudflare Worker (edge/worker.js) on the laptop, in front of a local API and
 * frontend, for the penetration test (scripts/pentest.py, Phase 8). Nothing in the Worker changes.
 *
 *   node scripts/edge_local.mjs --port 8787 --api http://127.0.0.1:8766 --frontend http://127.0.0.1:3000
 *
 * The secret comes from the EDGE_SHARED_SECRET environment variable, never the command line.
 * Two adapters stand in for what Cloudflare's runtime does and Node's doesn't:
 *   - cf-connecting-ip is set from the connection's own address, replacing anything the client
 *     sent, as Cloudflare does. A client can't choose its address here either.
 *   - Node's fetch needs `duplex: "half"` to send a streamed body; Workers don't.
 */
import http from "node:http";
import { parseArgs } from "node:util";
import worker from "../edge/worker.js";

const { values: args } = parseArgs({
  options: {
    port: { type: "string", default: "8787" },
    api: { type: "string", default: "http://127.0.0.1:8766" },
    frontend: { type: "string", default: "http://127.0.0.1:3000" },
  },
});
const env = {
  API_ORIGIN: args.api,
  FRONTEND_ORIGIN: args.frontend,
  EDGE_SHARED_SECRET: process.env.EDGE_SHARED_SECRET,
};

const nodeFetch = globalThis.fetch;
globalThis.fetch = (input, init = {}) =>
  nodeFetch(input, init.body ? { ...init, duplex: "half" } : init);

http
  .createServer(async (incoming, outgoing) => {
    try {
      const chunks = [];
      for await (const chunk of incoming) chunks.push(chunk);
      const headers = new Headers();
      for (const [name, value] of Object.entries(incoming.headers)) {
        if (name === "cf-connecting-ip" || name === "host" || name === "connection") continue;
        headers.set(name, Array.isArray(value) ? value.join(", ") : value);
      }
      headers.set("cf-connecting-ip", incoming.socket.remoteAddress.replace(/^::ffff:/, ""));
      const body = ["GET", "HEAD"].includes(incoming.method) ? undefined : Buffer.concat(chunks);
      const request = new Request(`http://localhost:${args.port}${incoming.url}`, {
        method: incoming.method,
        headers,
        body,
      });
      const response = await worker.fetch(request, env);
      const out = {};
      response.headers.forEach((value, name) => {
        if (name !== "set-cookie" && name !== "content-encoding" && name !== "content-length") {
          out[name] = value;
        }
      });
      const cookies = response.headers.getSetCookie();
      if (cookies.length) out["set-cookie"] = cookies;
      outgoing.writeHead(response.status, out);
      outgoing.end(Buffer.from(await response.arrayBuffer()));
    } catch (error) {
      outgoing.writeHead(502, { "content-type": "text/plain" });
      outgoing.end(`edge_local: ${error.message}`);
    }
  })
  .listen(Number(args.port), "127.0.0.1", () => {
    console.log(`Worker on http://localhost:${args.port} -> API ${args.api}, frontend ${args.frontend}`);
  });
