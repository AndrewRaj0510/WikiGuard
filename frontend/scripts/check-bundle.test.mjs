import { describe, expect, it } from "vitest";
import { findSecrets } from "./check-bundle.mjs";

describe("the client bundle secret scan", () => {
  it("finds real key shapes", () => {
    expect(findSecrets(`const k = "sk-proj-${"a".repeat(40)}"`)).toEqual(["OpenAI key"]);
    expect(findSecrets(`"AIza${"B".repeat(35)}"`)).toEqual(["Google API key"]);
    expect(findSecrets("postgresql://api:hunter2@db:5432/x")).toEqual(["Postgres URL with a password"]);
    expect(findSecrets("$argon2id$v=19$m=65536")).toEqual(["Argon2 hash"]);
    expect(findSecrets("-----BEGIN RSA PRIVATE KEY-----")).toEqual(["private key"]);
  });

  it("finds the value of a server-only setting", () => {
    expect(findSecrets("x=wikiguard-session-key-123", [["SESSION_SECRET_KEY", "wikiguard-session-key-123"]])).toEqual([
      "the value of SESSION_SECRET_KEY",
    ]);
  });

  it("ignores Tailwind's mask class names (F-117) and a Postgres URL with no password", () => {
    expect(findSecrets('{"mask-image-linear-from-pos":[{"mask-linear-from":1}]}')).toEqual([]);
    expect(findSecrets("postgresql://localhost/db")).toEqual([]);
  });
});
