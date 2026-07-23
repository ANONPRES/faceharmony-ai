# syntax=docker/dockerfile:1

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Browser calls same-origin /fh-api; Next rewrites to BACKEND_ORIGIN at runtime.
ARG NEXT_PUBLIC_API_URL=/fh-api
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
# Placeholder for build-time rewrite resolution; real value set at runtime.
ENV BACKEND_ORIGIN=http://127.0.0.1:8001

RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV NEXT_PUBLIC_API_URL=/fh-api
ENV BACKEND_ORIGIN=http://127.0.0.1:8001

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
