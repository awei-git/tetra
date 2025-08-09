<template>
  <nav class="bg-gray-800 shadow-lg border-b border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16">
        <div class="flex">
          <!-- Logo/Brand -->
          <div class="flex-shrink-0 flex items-center">
            <h2 class="text-xl font-bold text-gray-100">Tetra</h2>
          </div>
          
          <!-- Navigation Links -->
          <div class="hidden sm:ml-8 sm:flex sm:space-x-8">
            <router-link
              to="/"
              class="nav-link"
              :class="{ 'active': $route.path === '/' }"
            >
              Monitor
            </router-link>
            <router-link
              to="/chat"
              class="nav-link"
              :class="{ 'active': $route.path === '/chat' }"
            >
              Chat
            </router-link>
            <router-link
              to="/strategies"
              class="nav-link"
              :class="{ 'active': $route.path === '/strategies' }"
            >
              Strats
            </router-link>
          </div>
        </div>
        
        <!-- Right side items -->
        <div class="flex items-center space-x-4">
          <!-- Status indicator -->
          <div class="flex items-center space-x-2 text-sm">
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-gray-600">Connected</span>
          </div>
          
          <!-- Settings button -->
          <button class="p-2 text-gray-400 hover:text-gray-600">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>
      
      <!-- Mobile menu -->
      <div class="sm:hidden">
        <div class="pt-2 pb-3 space-y-1">
          <router-link
            to="/"
            class="mobile-nav-link"
            :class="{ 'active': $route.path === '/' }"
          >
            Monitor
          </router-link>
          <router-link
            to="/chat"
            class="mobile-nav-link"
            :class="{ 'active': $route.path === '/chat' }"
          >
            Chat
          </router-link>
          <router-link
            to="/strategies"
            class="mobile-nav-link"
            :class="{ 'active': $route.path === '/strategies' }"
          >
            Strats
          </router-link>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup>
// No additional logic needed for now
</script>

<style scoped>
.nav-link {
  @apply inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium;
  @apply border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500;
}

.nav-link.active {
  @apply border-blue-400 text-gray-100;
}

.mobile-nav-link {
  @apply block pl-3 pr-4 py-2 border-l-4 text-base font-medium;
  @apply border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-700 hover:border-gray-500;
}

.mobile-nav-link.active {
  @apply bg-gray-700 border-blue-400 text-blue-300;
}
</style>