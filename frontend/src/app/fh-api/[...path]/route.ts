import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function backendOrigin(): string {
  let origin = process.env.BACKEND_ORIGIN?.trim() || "http://127.0.0.1:8001";
  if (!/^https?:\/\//i.test(origin)) {
    origin = `http://${origin}`;
  }
  return origin.replace(/\/$/, "");
}

async function proxy(
  req: NextRequest,
  pathParts: string[],
): Promise<NextResponse> {
  const target = `${backendOrigin()}/${pathParts.join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const accept = req.headers.get("accept");
  if (accept) headers.set("accept", accept);

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: "manual",
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = Buffer.from(await req.arrayBuffer());
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (err) {
    const message = err instanceof Error ? err.message : "upstream error";
    return NextResponse.json(
      { detail: `Backend unreachable: ${message}` },
      { status: 502 },
    );
  }

  const outHeaders = new Headers();
  const pass = ["content-type", "cache-control"];
  for (const key of pass) {
    const value = upstream.headers.get(key);
    if (value) outHeaders.set(key, value);
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: outHeaders,
  });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PUT(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PATCH(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function DELETE(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
      "access-control-allow-headers": "content-type",
    },
  });
}
