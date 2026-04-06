-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Documents table
-- Stores document metadata per user
create table if not exists public.documents (
    id            text primary key,
    user_id       uuid references auth.users(id) on delete cascade not null,
    filename      text not null,
    title         text,
    authors       text[],
    status        text default 'processing',
    pages         integer,
    keywords      text[],
    abstract      text,
    year          integer,
    upload_date   timestamptz default now(),
    updated_at    timestamptz default now(),
    metadata      jsonb default '{}'::jsonb
);

-- Chat sessions table
-- Each conversation is a session
create table if not exists public.chat_sessions (
    id          uuid primary key default uuid_generate_v4(),
    user_id     uuid references auth.users(id) on delete cascade not null,
    title       text default 'New Conversation',
    created_at  timestamptz default now(),
    updated_at  timestamptz default now()
);

-- Chat messages table
-- Individual messages within a session
create table if not exists public.chat_messages (
    id          uuid primary key default uuid_generate_v4(),
    session_id  uuid references public.chat_sessions(id) on delete cascade not null,
    role        text not null check (role in ('user', 'assistant')),
    content     text not null,
    citations   jsonb default '[]'::jsonb,
    created_at  timestamptz default now()
);

-- Search history table
create table if not exists public.search_history (
    id            uuid primary key default uuid_generate_v4(),
    user_id       uuid references auth.users(id) on delete cascade not null,
    query         text not null,
    results_count integer default 0,
    created_at    timestamptz default now()
);

-- User profiles table (extends Supabase auth.users)
create table if not exists public.profiles (
    id            uuid primary key references auth.users(id) on delete cascade,
    full_name     text,
    avatar_url    text,
    created_at    timestamptz default now(),
    updated_at    timestamptz default now()
);

-- Auto-create profile on user signup
create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, full_name)
    values (new.id, new.raw_user_meta_data->>'full_name');
    return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();

-- Auto-update updated_at timestamps
create or replace function public.update_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger update_documents_updated_at
    before update on public.documents
    for each row execute procedure public.update_updated_at();

create trigger update_chat_sessions_updated_at
    before update on public.chat_sessions
    for each row execute procedure public.update_updated_at();

-- Row Level Security
alter table public.documents      enable row level security;
alter table public.chat_sessions  enable row level security;
alter table public.chat_messages  enable row level security;
alter table public.search_history enable row level security;
alter table public.profiles       enable row level security;

-- RLS Policies
create policy "users_own_documents"
    on public.documents for all
    using (auth.uid() = user_id);

create policy "users_own_sessions"
    on public.chat_sessions for all
    using (auth.uid() = user_id);

create policy "users_own_messages"
    on public.chat_messages for all
    using (session_id in (select id from public.chat_sessions where user_id = auth.uid()));

create policy "users_own_search_history"
    on public.search_history for all
    using (auth.uid() = user_id);

create policy "users_own_profile"
    on public.profiles for all
    using (auth.uid() = id);

-- Indexes for performance
create index if not exists idx_documents_user_id
    on public.documents(user_id);

create index if not exists idx_chat_sessions_user_id
    on public.chat_sessions(user_id);

create index if not exists idx_chat_messages_session_id
    on public.chat_messages(session_id);

create index if not exists idx_search_history_user_id
    on public.search_history(user_id);
