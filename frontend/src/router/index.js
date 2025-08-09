import { createRouter, createWebHistory } from 'vue-router'
import DataMonitor from '../views/DataMonitor.vue'
import ChatInterface from '../views/ChatInterface.vue'
import Strategies from '../views/Strategies.vue'

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
  },
  {
    path: '/strategies',
    name: 'Strategies',
    component: Strategies
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router