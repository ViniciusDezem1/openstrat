import Fastify, { FastifyInstance } from 'fastify'
import FastifySwagger from '@fastify/swagger'
import FastifySwaggerUi from '@fastify/swagger-ui'

// Instantiate the framework
const fastify: FastifyInstance = Fastify({
  logger: true,
})

// Set up @fastify/swagger
await fastify.register(FastifySwagger, {
  openapi: {
    info: {
      title: 'OpenStrat API',
      version: '1.0.0',
    },
    components: {
      securitySchemes: {
        apiKey: {
          type: 'apiKey',
          name: 'apiKey',
          in: 'header',
        },
      },
    },
  },
})

// Set up Swagger UI
await fastify.register(FastifySwaggerUi, {
  routePrefix: '/documentation',
  uiConfig: {
    docExpansion: 'full',
    deepLinking: false,
  },
  uiHooks: {
    onRequest: function (request, reply, next) {
      next()
    },
    preHandler: function (request, reply, next) {
      next()
    },
  },
  staticCSP: true,
  transformStaticCSP: (header) => header,
  transformSpecification: (swaggerObject, request, reply) => {
    return swaggerObject
  },
  transformSpecificationClone: true,
})

// Set up Scalar API Reference
await fastify.register(import('@scalar/fastify-api-reference'), {
  routePrefix: '/reference',
  configuration: {
    title: 'OpenStrat API Reference',
    theme: 'purple',
  },
})

// Declare a route
fastify.get('/', async function handler (request, reply) {
  return { hello: 'world' }
})

fastify.put(
  '/example-route/:id',
  {
    schema: {
      description: 'post some data',
      tags: ['user', 'code'],
      summary: 'qwerty',
      security: [{ apiKey: [] }],
      params: {
        type: 'object',
        properties: {
          id: {
            type: 'string',
            description: 'user id',
          },
        },
      },
      body: {
        type: 'object',
        properties: {
          hello: { type: 'string' },
          obj: {
            type: 'object',
            properties: {
              some: { type: 'string' },
            },
          },
        },
      },
      response: {
        201: {
          description: 'Successful response',
          type: 'object',
          properties: {
            hello: { type: 'string' },
          },
        },
        default: {
          description: 'Default response',
          type: 'object',
          properties: {
            foo: { type: 'string' },
          },
        },
      },
    },
  },
  (req, reply) => {
    const body = req.body as { hello: string }
    reply.code(201).send({ hello: `Hello ${body.hello}` })
  },
)

// Serve an OpenAPI file
fastify.get('/openapi.json', async (request, reply) => {
  return fastify.swagger()
})

// Wait for Fastify
await fastify.ready()

// Run the server
const start = async () => {
  try {
    await fastify.listen({ port: 3000 })
    console.log(`Fastify is now listening on http://localhost:3000`)
    console.log(`Swagger UI available at http://localhost:3000/documentation`)
    console.log(`Scalar API Reference available at http://localhost:3000/reference`)
  } catch (err) {
    fastify.log.error(err)
    process.exit(1)
  }
}

start()
