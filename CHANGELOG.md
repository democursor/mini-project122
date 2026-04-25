# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Supabase authentication system with JWT tokens
- User-scoped data isolation for documents and chat sessions
- Premium UI design with glass morphism effects
- Enhanced citation metadata with authors and publication year
- Neo4j fallback logic for graceful degradation
- Comprehensive README with setup instructions
- Frontend authentication context and protected routes
- Chat session persistence to database
- Search history tracking

### Changed
- Updated all API routes to require authentication
- Migrated chat storage from localStorage to Supabase database
- Enhanced frontend UI with modern design system
- Improved citation formatting with author names and years

### Security
- Implemented Row Level Security (RLS) policies in database
- Added JWT token validation middleware
- Secured all API endpoints with authentication

## [1.0.0] - 2024-03-22

### Added
- Initial release with core functionality
- PDF document processing pipeline
- Knowledge graph construction with Neo4j
- Semantic search with ChromaDB
- RAG-based chat assistant
- React frontend with multiple pages
- FastAPI backend with REST API
