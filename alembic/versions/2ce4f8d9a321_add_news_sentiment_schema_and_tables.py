"""Add news sentiment schema and tables

Revision ID: 2ce4f8d9a321
Revises: 9675352ad712
Create Date: 2025-08-01 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2ce4f8d9a321'
down_revision: Union[str, None] = '9675352ad712'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create news schema
    op.execute('CREATE SCHEMA IF NOT EXISTS news')
    
    # Create news_articles table
    op.create_table(
        'news_articles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_id', sa.String(length=500), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('source_category', sa.String(length=50), nullable=False),
        sa.Column('author', sa.String(length=200), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('image_url', sa.Text(), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbols', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source', 'source_id', name='uq_source_article'),
        schema='news'
    )
    
    # Create indexes for news_articles
    op.create_index('idx_news_published', 'news_articles', ['published_at'], schema='news')
    op.execute('CREATE INDEX idx_news_symbols ON news.news_articles USING gin (symbols jsonb_path_ops)')
    op.create_index('idx_news_source', 'news_articles', ['source'], schema='news')
    
    # Create news_sentiments table
    op.create_table(
        'news_sentiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('article_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('polarity', sa.Float(), nullable=False),
        sa.Column('subjectivity', sa.Float(), nullable=False),
        sa.Column('positive', sa.Float(), nullable=False),
        sa.Column('negative', sa.Float(), nullable=False),
        sa.Column('neutral', sa.Float(), nullable=False),
        sa.Column('bullish', sa.Float(), nullable=True),
        sa.Column('bearish', sa.Float(), nullable=True),
        sa.Column('sentiment_model', sa.String(length=100), nullable=False),
        sa.Column('analyzed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbols', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('impact_score', sa.Float(), nullable=True),
        sa.Column('is_breaking', sa.Boolean(), nullable=True),
        sa.Column('is_rumor', sa.Boolean(), nullable=True),
        sa.Column('requires_confirmation', sa.Boolean(), nullable=True),
        sa.Column('sentiment_by_type', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['article_id'], ['news.news_articles.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='news'
    )
    
    # Create indexes for news_sentiments
    op.create_index('idx_sentiment_article', 'news_sentiments', ['article_id'], schema='news')
    op.create_index('idx_sentiment_analyzed', 'news_sentiments', ['analyzed_at'], schema='news')
    op.execute('CREATE INDEX idx_sentiment_symbols ON news.news_sentiments USING gin (symbols jsonb_path_ops)')
    
    # Create news_clusters table
    op.create_table(
        'news_clusters',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('article_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('lead_article_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbols', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('earliest_published', sa.DateTime(timezone=True), nullable=False),
        sa.Column('latest_published', sa.DateTime(timezone=True), nullable=False),
        sa.Column('avg_polarity', sa.Float(), nullable=False),
        sa.Column('sentiment_std', sa.Float(), nullable=False),
        sa.Column('article_count', sa.Integer(), nullable=False),
        sa.Column('coherence_score', sa.Float(), nullable=False),
        sa.Column('velocity', sa.Float(), nullable=False),
        sa.Column('is_trending', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='news'
    )
    
    # Create indexes for news_clusters
    op.create_index('idx_cluster_created', 'news_clusters', ['created_at'], schema='news')
    op.execute('CREATE INDEX idx_cluster_symbols ON news.news_clusters USING gin (symbols jsonb_path_ops)')
    
    # Create news_summaries table
    op.create_table(
        'news_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('total_articles', sa.Integer(), nullable=False),
        sa.Column('positive_articles', sa.Integer(), nullable=False),
        sa.Column('negative_articles', sa.Integer(), nullable=False),
        sa.Column('neutral_articles', sa.Integer(), nullable=False),
        sa.Column('avg_sentiment', sa.Float(), nullable=False),
        sa.Column('sentiment_std', sa.Float(), nullable=False),
        sa.Column('article_velocity', sa.Float(), nullable=False),
        sa.Column('velocity_change', sa.Float(), nullable=False),
        sa.Column('top_categories', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('key_entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('most_positive_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('most_negative_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('most_impactful_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'start_date', 'end_date', name='uq_summary_period'),
        schema='news'
    )
    
    # Create indexes for news_summaries
    op.create_index('idx_summary_symbol_date', 'news_summaries', ['symbol', 'start_date'], schema='news')


def downgrade() -> None:
    # Drop tables
    op.drop_table('news_summaries', schema='news')
    op.drop_table('news_clusters', schema='news')
    op.drop_table('news_sentiments', schema='news')
    op.drop_table('news_articles', schema='news')
    
    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS news CASCADE')