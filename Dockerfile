FROM node:22-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

ENV NODE_ENV=production
ENV PORT=3000
ENV PYTHON_BIN=python3

EXPOSE 3000

CMD ["npm", "start"]
