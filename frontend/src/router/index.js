import { createRouter, createWebHistory } from 'vue-router'
import DataMonitor from '../views/DataMonitor.vue'
import ChatInterface from '../views/ChatInterface.vue'

const routes = [
  {
    path: '/',
    name: 'DataMonitor',
    component: DataMonitor
  },
  {
    path: '/chat',
    name: 'ChatInterface',
    component: ChatInterface
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router